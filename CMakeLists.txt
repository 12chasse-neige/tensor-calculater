cmake_minimum_required(VERSION 3.24)
project(TensorCalculatorCore VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(FetchContent)

# Pin the algebra engine and protocol dependency; optional heavyweight numerical
# backends are unnecessary for exact coordinate-component tensor calculations.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(WITH_SYMENGINE_THREAD_SAFE ON CACHE BOOL "" FORCE)
set(WITH_SYMENGINE_RCP ON CACHE BOOL "" FORCE)
set(WITH_LLVM OFF CACHE BOOL "" FORCE)
set(WITH_FLINT OFF CACHE BOOL "" FORCE)
set(WITH_OPENMP OFF CACHE BOOL "" FORCE)
set(INTEGER_CLASS gmp CACHE STRING "" FORCE)
FetchContent_Declare(symengine
  URL https://github.com/symengine/symengine/archive/refs/tags/v0.14.0.tar.gz
  URL_HASH SHA256=11c5f64e9eec998152437f288b8429ec001168277d55f3f5f1df78e3cf129707
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz
  URL_HASH SHA256=4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(symengine json)

add_subdirectory(core)
