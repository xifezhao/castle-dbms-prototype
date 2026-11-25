//! CASTLE Concurrency: Thread-Local `!Send` RCU Epoch Manager
//!
//! Each reader thread owns a private cache-line-aligned `ThreadEpochSlot`,
//! eliminating cross-core atomic contention on the read hot path while
//! guaranteeing memory-safe deferred reclamation on the background rebuild path.

#![deny(missing_docs)]

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

#[cfg(not(loom))]
use std::sync::atomic::AtomicUsize;
#[cfg(loom)]
use loom::sync::atomic::AtomicUsize;

const UNPINNED: usize = usize::MAX;

type DestructorFn = unsafe fn(*mut ());

struct GarbageItem {
    retired_epoch: usize,
    ptr: *mut (),
    destructor: DestructorFn,
}

unsafe impl Send for GarbageItem {}

#[repr(align(64))]
struct ThreadEpochSlot {
    epoch: AtomicUsize,
}

impl ThreadEpochSlot {
    fn new() -> Self {
        Self {
            epoch: AtomicUsize::new(UNPINNED),
        }
    }
}

/// Global epoch watermark and deferred garbage collector.
pub struct GlobalEpochManager {
    current_epoch: AtomicUsize,
    slots: Mutex<Vec<&'static ThreadEpochSlot>>,
    garbage: Mutex<Vec<GarbageItem>>,
}

#[cfg(not(loom))]
static GLOBAL_RCU: GlobalEpochManager = GlobalEpochManager {
    current_epoch: AtomicUsize::new(1),
    slots: Mutex::new(Vec::new()),
    garbage: Mutex::new(Vec::new()),
};

#[cfg(not(loom))]
fn register_thread_slot() -> &'static ThreadEpochSlot {
    let slot: &'static ThreadEpochSlot = Box::leak(Box::new(ThreadEpochSlot::new()));
    if let Ok(mut guard) = GLOBAL_RCU.slots.lock() {
        guard.push(slot);
    }
    slot
}

#[cfg(not(loom))]
thread_local! {
    static TL_SLOT: &'static ThreadEpochSlot = register_thread_slot();
    static TL_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Compile-time `!Send` RAII guard enforcing synchronous thread-local EBR pinning.
pub struct EpochGuard {
    _not_send: PhantomData<*const ()>,
}

impl EpochGuard {
    /// Enters a thread-local EBR critical section (touches only this thread's private cache line).
    #[inline(always)]
    pub fn enter() -> Self {
        Self::pin()
    }

    /// Pins the current thread to the active global epoch.
    #[inline(always)]
    pub fn pin() -> Self {
        #[cfg(not(loom))]
        {
            TL_DEPTH.with(|depth| {
                let d = depth.get();
                depth.set(d + 1);
                if d == 0 {
                    let e = GLOBAL_RCU.current_epoch.load(Ordering::Acquire);
                    TL_SLOT.with(|slot| slot.epoch.store(e, Ordering::Release));
                }
            });
        }
        Self {
            _not_send: PhantomData,
        }
    }
}

impl Drop for EpochGuard {
    #[inline(always)]
    fn drop(&mut self) {
        #[cfg(not(loom))]
        {
            TL_DEPTH.with(|depth| {
                let d = depth.get();
                if d <= 1 {
                    depth.set(0);
                    TL_SLOT.with(|slot| slot.epoch.store(UNPINNED, Ordering::Release));
                } else {
                    depth.set(d - 1);
                }
            });
        }
    }
}

/// Retires an obsolete snapshot pointer (`LeafSnap`) and reclaims items strictly older
/// than the minimum active thread-local epoch.
pub fn retire_snapshot<T>(ptr: *mut T) {
    if ptr.is_null() {
        return;
    }
    #[cfg(not(loom))]
    {
        unsafe fn drop_impl<T>(ptr: *mut ()) {
            drop(Box::from_raw(ptr as *mut T));
        }

        let retired_epoch = GLOBAL_RCU.current_epoch.fetch_add(1, Ordering::AcqRel);
        let item = GarbageItem {
            retired_epoch,
            ptr: ptr as *mut (),
            destructor: drop_impl::<T>,
        };

        // Compute minimum active epoch across all registered reader threads
        let mut min_active = GLOBAL_RCU.current_epoch.load(Ordering::Acquire);
        if let Ok(slots) = GLOBAL_RCU.slots.lock() {
            for slot in slots.iter() {
                let e = slot.epoch.load(Ordering::Acquire);
                if e < min_active {
                    min_active = e;
                }
            }
        }

        let mut to_drop = Vec::new();
        if let Ok(mut garbage) = GLOBAL_RCU.garbage.lock() {
            garbage.push(item);
            // Retain items whose retired_epoch >= min_active (still potentially visible)
            let mut i = 0;
            while i < garbage.len() {
                if garbage[i].retired_epoch < min_active {
                    to_drop.push(garbage.swap_remove(i));
                } else {
                    i += 1;
                }
            }
        }

        for g in to_drop {
            unsafe {
                (g.destructor)(g.ptr);
            }
        }
    }
    #[cfg(loom)]
    {
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}

/// Explicitly advances the global epoch watermark.
pub fn advance_global_watermark() {
    #[cfg(not(loom))]
    {
        GLOBAL_RCU.current_epoch.fetch_add(1, Ordering::SeqCst);
    }
}
