cmake_minimum_required(VERSION 3.15)
project(castle_baselines LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mfma -DNDEBUG -Wno-unused-parameter -Wno-sign-compare")

add_library(castle_ffi_bridge STATIC ffi_wrapper/src/baseline_bridge.cpp)
target_include_directories(castle_ffi_bridge PUBLIC
    ffi_wrapper/include
    /content/third_party/ALEX/src/core
    /content/third_party/LIPP/src/core
)

install(TARGETS castle_ffi_bridge ARCHIVE DESTINATION lib)
install(DIRECTORY ffi_wrapper/include/ DESTINATION include)
