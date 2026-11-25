// Copyright 2026 System Architecture & Database Kernel Team.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! CASTLE Storage & Cold-Track Offloading Engine
//!
//! This module implements the zero-copy persistent storage backend and eBPF-driven 
//! offloading mechanisms described in **Section 3.3 (The Cold Track: Asynchronous IO & eBPF Offload)**
//! of the VLDB 2027 paper.

#![deny(missing_docs)]

pub mod cold_track;
#[cfg(not(loom))]
pub mod io_uring_backend;

#[cfg(not(loom))]
pub use io_uring_backend::{AlignedBuffer, IoUringBackend};
