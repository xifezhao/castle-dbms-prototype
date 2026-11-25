#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

// Opaque PIMPL wrappers for genuine C++ baseline indexes.

struct AlexBaseline {
    struct Impl;
    std::unique_ptr<Impl> impl;
    AlexBaseline();
    ~AlexBaseline();
};

struct LippBaseline {
    struct Impl;
    std::unique_ptr<Impl> impl;
    LippBaseline();
    ~LippBaseline();
};

struct BTreeOlcBaseline {
    struct Impl;
    std::unique_ptr<Impl> impl;
    BTreeOlcBaseline();
    ~BTreeOlcBaseline();
};

struct DeltaLearnedBaseline {
    struct Impl;
    std::unique_ptr<Impl> impl;
    DeltaLearnedBaseline();
    ~DeltaLearnedBaseline();
};

struct AlexPartBaseline {
    struct Impl;
    std::unique_ptr<Impl> impl;
    AlexPartBaseline();
    ~AlexPartBaseline();
};

// ALEX (microsoft/ALEX + shared_mutex)
std::unique_ptr<AlexBaseline> new_alex();
void alex_bulk_load(const AlexBaseline& index, const uint64_t* keys, const uint64_t* vals, size_t n);
void alex_insert(const AlexBaseline& index, uint64_t key, uint64_t value);
bool alex_read(const AlexBaseline& index, uint64_t key, uint64_t& out_value);
size_t alex_range_scan(const AlexBaseline& index, uint64_t low, size_t limit);
size_t alex_memory_bytes(const AlexBaseline& index);

// LIPP (Jiacheng-WU/lipp + shared_mutex)
std::unique_ptr<LippBaseline> new_lipp();
void lipp_bulk_load(const LippBaseline& index, const uint64_t* keys, const uint64_t* vals, size_t n);
void lipp_insert(const LippBaseline& index, uint64_t key, uint64_t value);
bool lipp_read(const LippBaseline& index, uint64_t key, uint64_t& out_value);
size_t lipp_range_scan(const LippBaseline& index, uint64_t low, size_t limit);
size_t lipp_memory_bytes(const LippBaseline& index);

// B+Tree with Optimistic Lock Coupling (OLC)
std::unique_ptr<BTreeOlcBaseline> new_btree_olc();
void btree_olc_bulk_load(const BTreeOlcBaseline& index, const uint64_t* keys, const uint64_t* vals, size_t n);
void btree_olc_insert(const BTreeOlcBaseline& index, uint64_t key, uint64_t value);
bool btree_olc_read(const BTreeOlcBaseline& index, uint64_t key, uint64_t& out_value);
size_t btree_olc_range_scan(const BTreeOlcBaseline& index, uint64_t low, size_t limit);
size_t btree_olc_memory_bytes(const BTreeOlcBaseline& index);

// Delta-Buffered Learned Index (XIndex-style foreground-merge baseline)
std::unique_ptr<DeltaLearnedBaseline> new_delta_learned();
void delta_learned_bulk_load(const DeltaLearnedBaseline& index, const uint64_t* keys, const uint64_t* vals, size_t n);
void delta_learned_insert(const DeltaLearnedBaseline& index, uint64_t key, uint64_t value);
bool delta_learned_read(const DeltaLearnedBaseline& index, uint64_t key, uint64_t& out_value);
size_t delta_learned_range_scan(const DeltaLearnedBaseline& index, uint64_t low, size_t limit);
size_t delta_learned_memory_bytes(const DeltaLearnedBaseline& index);

// Range-Partitioned Concurrent ALEX (2,048 partitions, per-partition shared_mutex)
std::unique_ptr<AlexPartBaseline> new_alex_part();
void alex_part_bulk_load(const AlexPartBaseline& index, const uint64_t* keys, const uint64_t* vals, size_t n);
void alex_part_insert(const AlexPartBaseline& index, uint64_t key, uint64_t value);
bool alex_part_read(const AlexPartBaseline& index, uint64_t key, uint64_t& out_value);
size_t alex_part_range_scan(const AlexPartBaseline& index, uint64_t low, size_t limit);
size_t alex_part_memory_bytes(const AlexPartBaseline& index);
