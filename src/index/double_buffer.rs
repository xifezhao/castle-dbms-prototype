//! CASTLE Core: Lock-Free Double Buffering & Lazy On-Demand Allocation
//!
//! Implements the bounded lock-free append-only delta buffer (`OverflowBuffer`)
//! with lazy `OnceLock` slab allocation so read-only/cold leaves incur zero
//! delta-buffer heap overhead, while hot leaves allocate a compact 256-slot slab
//! on first write.

#![allow(unsafe_code)]

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::Ordering;
use std::sync::OnceLock;

#[cfg(not(loom))]
use std::sync::atomic::{AtomicBool, AtomicUsize};
#[cfg(loom)]
use loom::sync::atomic::{AtomicBool, AtomicUsize};

use crate::{IndexKey, IndexValue};

/// Compact capacity limit for a single delta buffer block (6 KiB per allocated slab).
pub const BUFFER_CAPACITY: usize = 256;

/// Error types for lock-free append operations.
#[derive(Debug, PartialEq, Eq)]
pub enum AtomicAppendError {
    /// Physical capacity of the buffer has been exhausted (backpressure signal).
    CapacityExhausted,
}

struct Slot<K, V> {
    key: UnsafeCell<MaybeUninit<K>>,
    value: UnsafeCell<MaybeUninit<V>>,
    ready: AtomicBool,
}

impl<K, V> Default for Slot<K, V> {
    fn default() -> Self {
        Self {
            key: UnsafeCell::new(MaybeUninit::uninit()),
            value: UnsafeCell::new(MaybeUninit::uninit()),
            ready: AtomicBool::new(false),
        }
    }
}

/// Lock-free append-only delta buffer with lazy on-demand slab allocation.
#[repr(align(64))]
pub struct OverflowBuffer<K: IndexKey, V: IndexValue> {
    tail: AtomicUsize,
    slots: OnceLock<Box<[Slot<K, V>]>>,
}

/// Type alias mapping `LevelBins` to `OverflowBuffer`.
pub type LevelBins<K, V> = OverflowBuffer<K, V>;

unsafe impl<K: IndexKey, V: IndexValue> Send for OverflowBuffer<K, V> {}
unsafe impl<K: IndexKey, V: IndexValue> Sync for OverflowBuffer<K, V> {}

impl<K: IndexKey, V: IndexValue> Default for OverflowBuffer<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: IndexKey, V: IndexValue> OverflowBuffer<K, V> {
    /// Creates an empty delta buffer with zero heap allocation until first write.
    pub fn new() -> Self {
        Self {
            tail: AtomicUsize::new(0),
            slots: OnceLock::new(),
        }
    }

    #[inline]
    fn get_or_init_slots(&self) -> &[Slot<K, V>] {
        self.slots.get_or_init(|| {
            let mut v = Vec::with_capacity(BUFFER_CAPACITY);
            for _ in 0..BUFFER_CAPACITY {
                v.push(Slot::default());
            }
            v.into_boxed_slice()
        })
    }

    /// Returns the approximate number of appended entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.tail.load(Ordering::Acquire).min(BUFFER_CAPACITY)
    }

    /// Returns true if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns actual heap + struct memory usage in bytes (0 slab bytes if never written).
    pub fn memory_bytes(&self) -> usize {
        let slab = self
            .slots
            .get()
            .map(|s| s.len() * std::mem::size_of::<Slot<K, V>>())
            .unwrap_or(0);
        std::mem::size_of::<Self>() + slab
    }

    /// Safely clears the buffer after an epoch rotation.
    pub fn clear(&self) {
        if let Some(slots) = self.slots.get() {
            let current_tail = self.tail.load(Ordering::Relaxed).min(BUFFER_CAPACITY);
            for i in 0..current_tail {
                slots[i].ready.store(false, Ordering::Relaxed);
            }
        }
        self.tail.store(0, Ordering::Release);
    }

    /// Wait-free append returning the allocated slot index.
    #[inline]
    pub fn append(&self, key: K, value: V) -> Result<usize, AtomicAppendError> {
        let slots = self.get_or_init_slots();
        let idx = self.tail.fetch_add(1, Ordering::Relaxed);
        if idx >= BUFFER_CAPACITY {
            self.tail.fetch_sub(1, Ordering::Relaxed);
            return Err(AtomicAppendError::CapacityExhausted);
        }
        let slot = &slots[idx];
        unsafe {
            (*slot.key.get()).write(key);
            (*slot.value.get()).write(value);
        }
        slot.ready.store(true, Ordering::Release);
        Ok(idx)
    }

    /// Non-blocking reverse scan to find the latest value for `search_key`.
    #[inline]
    pub fn scan(&self, search_key: &K) -> Option<V> {
        let scan_limit = self.tail.load(Ordering::Acquire).min(BUFFER_CAPACITY);
        if scan_limit == 0 {
            return None;
        }
        let slots = self.slots.get()?;
        for i in (0..scan_limit).rev() {
            let slot = &slots[i];
            if slot.ready.load(Ordering::Acquire) {
                unsafe {
                    let k = (*slot.key.get()).assume_init();
                    if k == *search_key {
                        return Some((*slot.value.get()).assume_init_ref().clone());
                    }
                }
            }
        }
        None
    }

    /// Collects all ready entries with key >= `low` for range scan merging.
    #[inline]
    pub fn scan_ge(&self, low: &K, out: &mut Vec<(K, V)>) {
        let scan_limit = self.tail.load(Ordering::Acquire).min(BUFFER_CAPACITY);
        if scan_limit == 0 {
            return;
        }
        let Some(slots) = self.slots.get() else {
            return;
        };
        for i in 0..scan_limit {
            let slot = &slots[i];
            if slot.ready.load(Ordering::Acquire) {
                unsafe {
                    let k = (*slot.key.get()).assume_init();
                    if k >= *low {
                        let v = (*slot.value.get()).assume_init_ref().clone();
                        out.push((k, v));
                    }
                }
            }
        }
    }

    /// Snapshots all ready entries in insertion order (called on frozen buffer during rebuild).
    pub fn snapshot_entries(&self) -> Vec<(K, V)> {
        let scan_limit = self.tail.load(Ordering::Acquire).min(BUFFER_CAPACITY);
        if scan_limit == 0 {
            return Vec::new();
        }
        let Some(slots) = self.slots.get() else {
            return Vec::new();
        };
        let mut entries = Vec::with_capacity(scan_limit);
        for i in 0..scan_limit {
            let slot = &slots[i];
            if slot.ready.load(Ordering::Acquire) {
                unsafe {
                    let k = (*slot.key.get()).assume_init();
                    let v = (*slot.value.get()).assume_init_ref().clone();
                    entries.push((k, v));
                }
            }
        }
        entries
    }
}
