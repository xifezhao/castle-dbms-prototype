#include "baseline_bridge.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <utility>
#include <vector>

// Official open-source header-only learned indexes
#include "alex.h"
#include "lipp.h"

// ============================================================================
// 1. ALEX (microsoft/ALEX guarded by reader-writer shared_mutex)
// ============================================================================
struct AlexBaseline::Impl {
    mutable std::shared_mutex mutex;
    alex::Alex<uint64_t, uint64_t> index;
};

AlexBaseline::AlexBaseline() : impl(std::make_unique<Impl>()) {}
AlexBaseline::~AlexBaseline() = default;

std::unique_ptr<AlexBaseline> new_alex() {
    return std::make_unique<AlexBaseline>();
}

void alex_bulk_load(const AlexBaseline& idx, const uint64_t* keys, const uint64_t* vals, size_t n) {
    std::unique_lock<std::shared_mutex> lock(idx.impl->mutex);
    std::vector<std::pair<uint64_t, uint64_t>> pairs;
    pairs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        pairs.emplace_back(keys[i], vals[i]);
    }
    idx.impl->index.bulk_load(pairs.data(), static_cast<int>(pairs.size()));
}

void alex_insert(const AlexBaseline& idx, uint64_t key, uint64_t value) {
    std::unique_lock<std::shared_mutex> lock(idx.impl->mutex);
    auto it = idx.impl->index.find(key);
    if (it != idx.impl->index.end()) {
        it.payload() = value;
    } else {
        idx.impl->index.insert(key, value);
    }
}

bool alex_read(const AlexBaseline& idx, uint64_t key, uint64_t& out_value) {
    std::shared_lock<std::shared_mutex> lock(idx.impl->mutex);
    auto* payload = idx.impl->index.get_payload(key);
    if (payload != nullptr) {
        out_value = *payload;
        return true;
    }
    return false;
}

size_t alex_range_scan(const AlexBaseline& idx, uint64_t low, size_t limit) {
    std::shared_lock<std::shared_mutex> lock(idx.impl->mutex);
    auto it = idx.impl->index.lower_bound(low);
    size_t count = 0;
    uint64_t acc = 0;
    while (it != idx.impl->index.end() && count < limit) {
        acc ^= it.payload();
        ++it;
        ++count;
    }
    (void)acc;
    return count;
}

size_t alex_memory_bytes(const AlexBaseline& idx) {
    std::shared_lock<std::shared_mutex> lock(idx.impl->mutex);
    return static_cast<size_t>(idx.impl->index.model_size() + idx.impl->index.data_size());
}

// ============================================================================
// 2. LIPP (Jiacheng-WU/lipp guarded by reader-writer shared_mutex)
// ============================================================================
struct LippBaseline::Impl {
    mutable std::shared_mutex mutex;
    LIPP<uint64_t, uint64_t> index;
};

LippBaseline::LippBaseline() : impl(std::make_unique<Impl>()) {}
LippBaseline::~LippBaseline() = default;

std::unique_ptr<LippBaseline> new_lipp() {
    return std::make_unique<LippBaseline>();
}

void lipp_bulk_load(const LippBaseline& idx, const uint64_t* keys, const uint64_t* vals, size_t n) {
    std::unique_lock<std::shared_mutex> lock(idx.impl->mutex);
    std::vector<std::pair<uint64_t, uint64_t>> pairs;
    pairs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        pairs.emplace_back(keys[i], vals[i]);
    }
    idx.impl->index.bulk_load(pairs.data(), static_cast<int>(pairs.size()));
}

void lipp_insert(const LippBaseline& idx, uint64_t key, uint64_t value) {
    std::unique_lock<std::shared_mutex> lock(idx.impl->mutex);
    if (!idx.impl->index.exists(key)) {
        idx.impl->index.insert(key, value);
    }
}

bool lipp_read(const LippBaseline& idx, uint64_t key, uint64_t& out_value) {
    std::shared_lock<std::shared_mutex> lock(idx.impl->mutex);
    if (idx.impl->index.exists(key)) {
        out_value = idx.impl->index.at(key, true);
        return true;
    }
    return false;
}

size_t lipp_range_scan(const LippBaseline& idx, uint64_t low, size_t limit) {
    std::shared_lock<std::shared_mutex> lock(idx.impl->mutex);
    size_t found = 0;
    uint64_t acc = 0;
    // LIPP's official core header exposes point lookup (exists/at) without a leaf-linked iterator;
    // emulate bounded short scan by probing keys in the target window.
    for (size_t offset = 0; offset < limit * 8 && found < limit; ++offset) {
        uint64_t probe = low + offset;
        if (idx.impl->index.exists(probe)) {
            acc ^= idx.impl->index.at(probe, true);
            ++found;
        }
    }
    (void)acc;
    return found;
}

size_t lipp_memory_bytes(const LippBaseline& idx) {
    std::shared_lock<std::shared_mutex> lock(idx.impl->mutex);
    return static_cast<size_t>(idx.impl->index.index_size(true));
}

// ============================================================================
// 3. Cache-Aligned B+Tree with Optimistic Lock Coupling (OLC)
// ============================================================================
namespace btree_olc {

constexpr int LEAF_CAP = 64;
constexpr int INNER_CAP = 64;

struct alignas(64) OptLock {
    std::atomic<uint64_t> version{0};

    static bool is_locked(uint64_t v) { return (v & 0b10) != 0; }
    static bool is_obsolete(uint64_t v) { return (v & 0b01) != 0; }

    uint64_t read_lock_or_restart(bool& need_restart) const {
        uint64_t v = version.load(std::memory_order_acquire);
        if (is_locked(v) || is_obsolete(v)) {
            need_restart = true;
        }
        return v;
    }

    void check_or_restart(uint64_t start_v, bool& need_restart) const {
        if (start_v != version.load(std::memory_order_acquire)) {
            need_restart = true;
        }
    }

    void write_lock_or_restart(bool& need_restart) {
        uint64_t v = version.load(std::memory_order_acquire);
        if (is_locked(v) || is_obsolete(v)) {
            need_restart = true;
            return;
        }
        if (!version.compare_exchange_strong(v, v + 0b10, std::memory_order_acq_rel)) {
            need_restart = true;
        }
    }

    void write_unlock() {
        version.fetch_add(0b10, std::memory_order_release);
    }

    void write_unlock_obsolete() {
        version.fetch_add(0b11, std::memory_order_release);
    }
};

struct NodeBase : OptLock {
    bool is_leaf;
    uint16_t count{0};
    explicit NodeBase(bool leaf) : is_leaf(leaf) {}
};

struct alignas(64) LeafNode : NodeBase {
    uint64_t keys[LEAF_CAP];
    uint64_t vals[LEAF_CAP];
    LeafNode* next{nullptr};
    LeafNode() : NodeBase(true) {}
};

struct alignas(64) InnerNode : NodeBase {
    uint64_t keys[INNER_CAP];
    NodeBase* children[INNER_CAP + 1];
    InnerNode() : NodeBase(false) {}
};

class BTreeOLC {
public:
    std::atomic<NodeBase*> root;
    std::atomic<size_t> node_count{1};

    BTreeOLC() {
        root.store(new LeafNode(), std::memory_order_release);
    }

    ~BTreeOLC() {
        free_subtree(root.load(std::memory_order_relaxed));
    }

    void free_subtree(NodeBase* n) {
        if (!n) return;
        if (!n->is_leaf) {
            auto* inner = static_cast<InnerNode*>(n);
            for (int i = 0; i <= inner->count; ++i) {
                free_subtree(inner->children[i]);
            }
            delete inner;
        } else {
            delete static_cast<LeafNode*>(n);
        }
    }

    static int lower_bound_u64(const uint64_t* arr, int n, uint64_t k) {
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (arr[mid] < k) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    bool lookup(uint64_t k, uint64_t& out_val) const {
        while (true) {
            bool restart = false;
            NodeBase* node = root.load(std::memory_order_acquire);
            uint64_t v = node->read_lock_or_restart(restart);
            if (restart) continue;

            while (!node->is_leaf) {
                auto* inner = static_cast<InnerNode*>(node);
                int pos = lower_bound_u64(inner->keys, inner->count, k);
                if (pos < inner->count && inner->keys[pos] == k) ++pos;
                NodeBase* next = inner->children[pos];
                node->check_or_restart(v, restart);
                if (restart) break;

                uint64_t next_v = next->read_lock_or_restart(restart);
                if (restart) break;
                node->check_or_restart(v, restart);
                if (restart) break;

                node = next;
                v = next_v;
            }
            if (restart) continue;

            auto* leaf = static_cast<LeafNode*>(node);
            int pos = lower_bound_u64(leaf->keys, leaf->count, k);
            bool found = (pos < leaf->count && leaf->keys[pos] == k);
            uint64_t val = found ? leaf->vals[pos] : 0;
            leaf->check_or_restart(v, restart);
            if (restart) continue;

            if (found) out_val = val;
            return found;
        }
    }

    size_t range_scan(uint64_t low, size_t limit) const {
        while (true) {
            bool restart = false;
            NodeBase* node = root.load(std::memory_order_acquire);
            uint64_t v = node->read_lock_or_restart(restart);
            if (restart) continue;

            while (!node->is_leaf) {
                auto* inner = static_cast<InnerNode*>(node);
                int pos = lower_bound_u64(inner->keys, inner->count, low);
                if (pos < inner->count && inner->keys[pos] == low) ++pos;
                NodeBase* next = inner->children[pos];
                node->check_or_restart(v, restart);
                if (restart) break;
                uint64_t next_v = next->read_lock_or_restart(restart);
                if (restart) break;
                node->check_or_restart(v, restart);
                if (restart) break;
                node = next;
                v = next_v;
            }
            if (restart) continue;

            auto* leaf = static_cast<LeafNode*>(node);
            size_t scanned = 0;
            uint64_t acc = 0;
            int pos = lower_bound_u64(leaf->keys, leaf->count, low);
            while (leaf && scanned < limit) {
                for (int i = pos; i < leaf->count && scanned < limit; ++i) {
                    acc ^= leaf->vals[i];
                    ++scanned;
                }
                LeafNode* nxt = leaf->next;
                leaf->check_or_restart(v, restart);
                if (restart) break;
                if (!nxt || scanned >= limit) break;
                v = nxt->read_lock_or_restart(restart);
                if (restart) break;
                leaf = nxt;
                pos = 0;
            }
            if (restart) continue;
            (void)acc;
            return scanned;
        }
    }

    void insert(uint64_t k, uint64_t val) {
        while (true) {
            bool restart = false;
            NodeBase* node = root.load(std::memory_order_acquire);
            uint64_t v = node->read_lock_or_restart(restart);
            if (restart) continue;

            InnerNode* parent = nullptr;
            uint64_t parent_v = 0;

            while (!node->is_leaf) {
                auto* inner = static_cast<InnerNode*>(node);
                if (inner->count == INNER_CAP) {
                    // Proactive split of full inner node
                    if (parent) {
                        parent->write_lock_or_restart(restart);
                        if (restart) break;
                        parent->check_or_restart(parent_v + 0b10, restart);
                        if (restart) { parent->write_unlock(); break; }
                    }
                    inner->write_lock_or_restart(restart);
                    if (restart) {
                        if (parent) parent->write_unlock();
                        break;
                    }
                    split_inner(parent, inner);
                    inner->write_unlock();
                    if (parent) parent->write_unlock();
                    restart = true;
                    break;
                }

                int pos = lower_bound_u64(inner->keys, inner->count, k);
                if (pos < inner->count && inner->keys[pos] == k) ++pos;
                NodeBase* next = inner->children[pos];
                inner->check_or_restart(v, restart);
                if (restart) break;
                uint64_t next_v = next->read_lock_or_restart(restart);
                if (restart) break;
                inner->check_or_restart(v, restart);
                if (restart) break;

                parent = inner;
                parent_v = v;
                node = next;
                v = next_v;
            }
            if (restart) continue;

            auto* leaf = static_cast<LeafNode*>(node);
            if (leaf->count == LEAF_CAP) {
                if (parent) {
                    parent->write_lock_or_restart(restart);
                    if (restart) continue;
                }
                leaf->write_lock_or_restart(restart);
                if (restart) {
                    if (parent) parent->write_unlock();
                    continue;
                }
                if (leaf->count == LEAF_CAP) {
                    split_leaf(parent, leaf);
                }
                leaf->write_unlock();
                if (parent) parent->write_unlock();
                continue;
            }

            leaf->write_lock_or_restart(restart);
            if (restart) continue;
            if (leaf->count == LEAF_CAP) {
                leaf->write_unlock();
                continue;
            }

            int pos = lower_bound_u64(leaf->keys, leaf->count, k);
            if (pos < leaf->count && leaf->keys[pos] == k) {
                leaf->vals[pos] = val;
            } else {
                for (int i = leaf->count; i > pos; --i) {
                    leaf->keys[i] = leaf->keys[i - 1];
                    leaf->vals[i] = leaf->vals[i - 1];
                }
                leaf->keys[pos] = k;
                leaf->vals[pos] = val;
                leaf->count++;
            }
            leaf->write_unlock();
            return;
        }
    }

    size_t memory_bytes() const {
        return node_count.load(std::memory_order_relaxed) * sizeof(LeafNode);
    }

private:
    void split_leaf(InnerNode* parent, LeafNode* leaf) {
        auto* sibling = new LeafNode();
        node_count.fetch_add(1, std::memory_order_relaxed);
        int mid = leaf->count / 2;
        int right_n = leaf->count - mid;
        for (int i = 0; i < right_n; ++i) {
            sibling->keys[i] = leaf->keys[mid + i];
            sibling->vals[i] = leaf->vals[mid + i];
        }
        sibling->count = static_cast<uint16_t>(right_n);
        sibling->next = leaf->next;
        leaf->next = sibling;
        leaf->count = static_cast<uint16_t>(mid);
        uint64_t sep = sibling->keys[0];

        if (!parent) {
            auto* new_root = new InnerNode();
            node_count.fetch_add(1, std::memory_order_relaxed);
            new_root->keys[0] = sep;
            new_root->children[0] = leaf;
            new_root->children[1] = sibling;
            new_root->count = 1;
            root.store(new_root, std::memory_order_release);
        } else {
            insert_into_inner(parent, sep, sibling);
        }
    }

    void split_inner(InnerNode* parent, InnerNode* inner) {
        auto* sibling = new InnerNode();
        node_count.fetch_add(1, std::memory_order_relaxed);
        int mid = inner->count / 2;
        uint64_t sep = inner->keys[mid];
        int right_n = inner->count - mid - 1;
        for (int i = 0; i < right_n; ++i) {
            sibling->keys[i] = inner->keys[mid + 1 + i];
            sibling->children[i] = inner->children[mid + 1 + i];
        }
        sibling->children[right_n] = inner->children[inner->count];
        sibling->count = static_cast<uint16_t>(right_n);
        inner->count = static_cast<uint16_t>(mid);

        if (!parent) {
            auto* new_root = new InnerNode();
            node_count.fetch_add(1, std::memory_order_relaxed);
            new_root->keys[0] = sep;
            new_root->children[0] = inner;
            new_root->children[1] = sibling;
            new_root->count = 1;
            root.store(new_root, std::memory_order_release);
        } else {
            insert_into_inner(parent, sep, sibling);
        }
    }

    static void insert_into_inner(InnerNode* parent, uint64_t sep, NodeBase* right_child) {
        int pos = lower_bound_u64(parent->keys, parent->count, sep);
        for (int i = parent->count; i > pos; --i) {
            parent->keys[i] = parent->keys[i - 1];
            parent->children[i + 1] = parent->children[i];
        }
        parent->keys[pos] = sep;
        parent->children[pos + 1] = right_child;
        parent->count++;
    }
};

} // namespace btree_olc

struct BTreeOlcBaseline::Impl {
    btree_olc::BTreeOLC tree;
};

BTreeOlcBaseline::BTreeOlcBaseline() : impl(std::make_unique<Impl>()) {}
BTreeOlcBaseline::~BTreeOlcBaseline() = default;

std::unique_ptr<BTreeOlcBaseline> new_btree_olc() {
    return std::make_unique<BTreeOlcBaseline>();
}

void btree_olc_bulk_load(const BTreeOlcBaseline& idx, const uint64_t* keys, const uint64_t* vals, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        idx.impl->tree.insert(keys[i], vals[i]);
    }
}

void btree_olc_insert(const BTreeOlcBaseline& idx, uint64_t key, uint64_t value) {
    idx.impl->tree.insert(key, value);
}

bool btree_olc_read(const BTreeOlcBaseline& idx, uint64_t key, uint64_t& out_value) {
    return idx.impl->tree.lookup(key, out_value);
}

size_t btree_olc_range_scan(const BTreeOlcBaseline& idx, uint64_t low, size_t limit) {
    return idx.impl->tree.range_scan(low, limit);
}

size_t btree_olc_memory_bytes(const BTreeOlcBaseline& idx) {
    return idx.impl->tree.memory_bytes();
}

// ============================================================================
// 4. Delta-Buffered Learned Index (XIndex-style foreground-merge baseline)
// ============================================================================
struct DeltaLearnedBaseline::Impl {
    static constexpr size_t NUM_GROUPS = 2048;
    static constexpr size_t DELTA_CAP = 512;

    struct alignas(64) Group {
        mutable std::shared_mutex mu;
        double slope{0.0};
        double intercept{0.0};
        std::vector<std::pair<uint64_t, uint64_t>> base;
        std::vector<std::pair<uint64_t, uint64_t>> delta;

        void fit_model() {
            size_t n = base.size();
            if (n <= 1) {
                slope = 0.0;
                intercept = 0.0;
                return;
            }
            double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double x = static_cast<double>(base[i].first);
                double y = static_cast<double>(i);
                sx += x; sy += y; sxx += x * x; sxy += x * y;
            }
            double nd = static_cast<double>(n);
            double denom = nd * sxx - sx * sx;
            if (std::abs(denom) < 1e-12) {
                slope = 0.0;
                intercept = sy / nd;
            } else {
                slope = (nd * sxy - sx * sy) / denom;
                intercept = (sy - slope * sx) / nd;
            }
        }

        void foreground_merge() {
            if (delta.empty()) return;
            std::stable_sort(delta.begin(), delta.end(),
                             [](const auto& a, const auto& b) { return a.first < b.first; });
            std::vector<std::pair<uint64_t, uint64_t>> merged;
            merged.reserve(base.size() + delta.size());
            size_t i = 0, j = 0;
            while (i < base.size() && j < delta.size()) {
                if (base[i].first < delta[j].first) {
                    merged.push_back(base[i++]);
                } else if (delta[j].first < base[i].first) {
                    if (!merged.empty() && merged.back().first == delta[j].first)
                        merged.back().second = delta[j].second;
                    else
                        merged.push_back(delta[j]);
                    ++j;
                } else {
                    if (!merged.empty() && merged.back().first == delta[j].first)
                        merged.back().second = delta[j].second;
                    else
                        merged.push_back(delta[j]);
                    ++i; ++j;
                }
            }
            while (i < base.size()) merged.push_back(base[i++]);
            while (j < delta.size()) {
                if (!merged.empty() && merged.back().first == delta[j].first)
                    merged.back().second = delta[j].second;
                else
                    merged.push_back(delta[j]);
                ++j;
            }
            base.swap(merged);
            delta.clear();
            fit_model();
        }
    };

    uint64_t key_min{0};
    uint64_t key_span{10000000};
    std::vector<Group> groups;

    Impl() : groups(NUM_GROUPS) {}

    size_t route(uint64_t k) const {
        if (k <= key_min) return 0;
        uint64_t rel = k - key_min;
        if (rel >= key_span) return NUM_GROUPS - 1;
        return static_cast<size_t>((static_cast<__uint128_t>(rel) * NUM_GROUPS) / key_span);
    }
};

DeltaLearnedBaseline::DeltaLearnedBaseline() : impl(std::make_unique<Impl>()) {}
DeltaLearnedBaseline::~DeltaLearnedBaseline() = default;

std::unique_ptr<DeltaLearnedBaseline> new_delta_learned() {
    return std::make_unique<DeltaLearnedBaseline>();
}

void delta_learned_bulk_load(const DeltaLearnedBaseline& idx, const uint64_t* keys, const uint64_t* vals, size_t n) {
    if (n == 0) return;
    idx.impl->key_min = keys[0];
    idx.impl->key_span = std::max<uint64_t>(1, keys[n - 1] - keys[0] + 1);
    for (size_t i = 0; i < n; ++i) {
        size_t g = idx.impl->route(keys[i]);
        idx.impl->groups[g].base.emplace_back(keys[i], vals[i]);
    }
    for (auto& g : idx.impl->groups) {
        g.fit_model();
    }
}

void delta_learned_insert(const DeltaLearnedBaseline& idx, uint64_t key, uint64_t value) {
    auto& g = idx.impl->groups[idx.impl->route(key)];
    std::unique_lock<std::shared_mutex> lock(g.mu);
    g.delta.emplace_back(key, value);
    if (g.delta.size() >= DeltaLearnedBaseline::Impl::DELTA_CAP) {
        g.foreground_merge();
    }
}

bool delta_learned_read(const DeltaLearnedBaseline& idx, uint64_t key, uint64_t& out_value) {
    const auto& g = idx.impl->groups[idx.impl->route(key)];
    std::shared_lock<std::shared_mutex> lock(g.mu);
    for (auto it = g.delta.rbegin(); it != g.delta.rend(); ++it) {
        if (it->first == key) {
            out_value = it->second;
            return true;
        }
    }
    if (g.base.empty()) return false;
    double raw = g.slope * static_cast<double>(key) + g.intercept;
    size_t pred = raw <= 0.0 ? 0 : std::min(static_cast<size_t>(raw), g.base.size() - 1);
    size_t lo = pred > 256 ? pred - 256 : 0;
    size_t hi = std::min(g.base.size(), pred + 257);
    auto it = std::lower_bound(g.base.begin() + lo, g.base.begin() + hi, key,
                               [](const auto& p, uint64_t k) { return p.first < k; });
    if (it != g.base.begin() + hi && it->first == key) {
        out_value = it->second;
        return true;
    }
    it = std::lower_bound(g.base.begin(), g.base.end(), key,
                          [](const auto& p, uint64_t k) { return p.first < k; });
    if (it != g.base.end() && it->first == key) {
        out_value = it->second;
        return true;
    }
    return false;
}

size_t delta_learned_range_scan(const DeltaLearnedBaseline& idx, uint64_t low, size_t limit) {
    size_t g_idx = idx.impl->route(low);
    size_t scanned = 0;
    uint64_t acc = 0;
    while (g_idx < DeltaLearnedBaseline::Impl::NUM_GROUPS && scanned < limit) {
        const auto& g = idx.impl->groups[g_idx];
        std::shared_lock<std::shared_mutex> lock(g.mu);
        auto it = std::lower_bound(g.base.begin(), g.base.end(), low,
                                   [](const auto& p, uint64_t k) { return p.first < k; });
        while (it != g.base.end() && scanned < limit) {
            acc ^= it->second;
            ++it;
            ++scanned;
        }
        ++g_idx;
    }
    (void)acc;
    return scanned;
}

size_t delta_learned_memory_bytes(const DeltaLearnedBaseline& idx) {
    size_t total = sizeof(DeltaLearnedBaseline::Impl);
    for (const auto& g : idx.impl->groups) {
        total += sizeof(DeltaLearnedBaseline::Impl::Group)
               + g.base.capacity() * sizeof(std::pair<uint64_t, uint64_t>)
               + g.delta.capacity() * sizeof(std::pair<uint64_t, uint64_t>);
    }
    return total;
}

// ============================================================================
// 5. Range-Partitioned Concurrent ALEX (2,048 partitions, per-partition RWLock)
// ============================================================================
struct AlexPartBaseline::Impl {
    static constexpr size_t NUM_PARTS = 2048;

    struct alignas(64) Part {
        mutable std::shared_mutex mu;
        alex::Alex<uint64_t, uint64_t> index;
    };

    uint64_t key_min{0};
    uint64_t key_span{10000000};
    std::vector<Part> parts;

    Impl() : parts(NUM_PARTS) {}

    size_t route(uint64_t k) const {
        if (k <= key_min) return 0;
        uint64_t rel = k - key_min;
        if (rel >= key_span) return NUM_PARTS - 1;
        return static_cast<size_t>((static_cast<__uint128_t>(rel) * NUM_PARTS) / key_span);
    }
};

AlexPartBaseline::AlexPartBaseline() : impl(std::make_unique<Impl>()) {}
AlexPartBaseline::~AlexPartBaseline() = default;

std::unique_ptr<AlexPartBaseline> new_alex_part() {
    return std::make_unique<AlexPartBaseline>();
}

void alex_part_bulk_load(const AlexPartBaseline& idx, const uint64_t* keys, const uint64_t* vals, size_t n) {
    if (n == 0) return;
    idx.impl->key_min = keys[0];
    idx.impl->key_span = std::max<uint64_t>(1, keys[n - 1] - keys[0] + 1);
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> buckets(AlexPartBaseline::Impl::NUM_PARTS);
    for (size_t i = 0; i < n; ++i) {
        buckets[idx.impl->route(keys[i])].emplace_back(keys[i], vals[i]);
    }
    for (size_t p = 0; p < AlexPartBaseline::Impl::NUM_PARTS; ++p) {
        if (!buckets[p].empty()) {
            std::unique_lock<std::shared_mutex> lock(idx.impl->parts[p].mu);
            idx.impl->parts[p].index.bulk_load(buckets[p].data(), static_cast<int>(buckets[p].size()));
        }
    }
}

void alex_part_insert(const AlexPartBaseline& idx, uint64_t key, uint64_t value) {
    auto& part = idx.impl->parts[idx.impl->route(key)];
    std::unique_lock<std::shared_mutex> lock(part.mu);
    auto it = part.index.find(key);
    if (it != part.index.end()) {
        it.payload() = value;
    } else {
        part.index.insert(key, value);
    }
}

bool alex_part_read(const AlexPartBaseline& idx, uint64_t key, uint64_t& out_value) {
    const auto& part = idx.impl->parts[idx.impl->route(key)];
    std::shared_lock<std::shared_mutex> lock(part.mu);
    auto* payload = part.index.get_payload(key);
    if (payload != nullptr) {
        out_value = *payload;
        return true;
    }
    return false;
}

size_t alex_part_range_scan(const AlexPartBaseline& idx, uint64_t low, size_t limit) {
    size_t p_idx = idx.impl->route(low);
    size_t scanned = 0;
    uint64_t acc = 0;
    while (p_idx < AlexPartBaseline::Impl::NUM_PARTS && scanned < limit) {
        auto& part = idx.impl->parts[p_idx];
        std::shared_lock<std::shared_mutex> lock(part.mu);
        auto it = part.index.lower_bound(low);
        while (it != part.index.end() && scanned < limit) {
            acc ^= it.payload();
            ++it;
            ++scanned;
        }
        ++p_idx;
    }
    (void)acc;
    return scanned;
}

size_t alex_part_memory_bytes(const AlexPartBaseline& idx) {
    size_t total = sizeof(AlexPartBaseline::Impl);
    for (const auto& part : idx.impl->parts) {
        std::shared_lock<std::shared_mutex> lock(part.mu);
        total += sizeof(AlexPartBaseline::Impl::Part)
               + static_cast<size_t>(part.index.model_size() + part.index.data_size());
    }
    return total;
}
