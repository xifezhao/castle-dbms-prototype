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

//! CASTLE Benchmark Suites (YCSB, TPC-C, Skew Burst)

pub mod ycsb {
    //! YCSB Workloads (A, C)
    pub mod workload_a;
    pub mod workload_c;
}

pub mod tpcc {
    //! TPC-C Workload
    pub mod index_stressed_tpcc;
}

pub mod micro {
    //! Skew Burst Microbenchmark
    pub mod skew_burst;
}
