cmake_minimum_required(VERSION 3.22)
project(speculative_prefetcher LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(LIBURING_PREFIX "" CACHE PATH "Path to liburing install prefix")
if(NOT LIBURING_PREFIX)
  set(LIBURING_PREFIX "/data/disk2/ljc/solidattention_deps/liburing-install")
endif()

find_package(CUDAToolkit REQUIRED)

find_path(LIBURING_INCLUDE_DIR liburing.h HINTS "${LIBURING_PREFIX}/include")
find_library(LIBURING_LIBRARY uring HINTS "${LIBURING_PREFIX}/lib")
if(NOT LIBURING_INCLUDE_DIR OR NOT LIBURING_LIBRARY)
  message(FATAL_ERROR "liburing not found under ${LIBURING_PREFIX}")
endif()

add_executable(speculative_prefetch_replay src/speculative_prefetch_replay.cu)
target_include_directories(speculative_prefetch_replay PRIVATE "${LIBURING_INCLUDE_DIR}")
target_link_libraries(speculative_prefetch_replay PRIVATE "${LIBURING_LIBRARY}" CUDA::cudart)
target_compile_options(speculative_prefetch_replay PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
set_target_properties(speculative_prefetch_replay PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_ARCHITECTURES 89
  BUILD_RPATH "${LIBURING_PREFIX}/lib"
)
