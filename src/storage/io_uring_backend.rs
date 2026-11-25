#![cfg(not(loom))]
//! CASTLE Direct-I/O Block Storage Backend
//!
//! Provides page-aligned memory buffers (`AlignedBuffer`) and positioned block reads.

#![deny(missing_docs)]

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;

const DIRECT_IO_ALIGNMENT: usize = 4096;

/// Page-aligned memory buffer for block-aligned storage reads.
pub struct AlignedBuffer {
    ptr: *mut u8,
    layout: std::alloc::Layout,
    capacity: usize,
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocates a new 4096-byte aligned buffer.
    pub fn new(size: usize) -> Self {
        let capacity = (size.max(1) + DIRECT_IO_ALIGNMENT - 1) & !(DIRECT_IO_ALIGNMENT - 1);
        let layout = std::alloc::Layout::from_size_align(capacity, DIRECT_IO_ALIGNMENT)
            .expect("Failed to create aligned memory layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self { ptr, layout, capacity }
    }

    /// Returns a mutable slice into the aligned buffer.
    ///
    /// # Safety
    /// Caller must ensure exclusive access while writing.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.ptr, self.capacity)
    }

    /// Returns an immutable slice of the aligned buffer.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.capacity) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.ptr, self.layout);
        }
    }
}

/// Positioned block storage backend for cold-track reads.
pub struct IoUringBackend {
    file: File,
}

impl IoUringBackend {
    /// Opens a backing file for positioned block reads.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self { file })
    }

    /// Reads an aligned block at `offset` into `buf`.
    pub fn read_at(&self, buf: &mut AlignedBuffer, offset: u64) -> io::Result<usize> {
        let slice = unsafe { buf.as_mut_slice() };
        self.file.read_at(slice, offset)
    }
}
