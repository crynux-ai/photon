cmake_minimum_required(VERSION 3.11)
project("photon" CXX)

set(CMAKE_CXX_STANDARD 20)


set(PLATFORM "CPU" CACHE STRING "Select an option: CPU, METAL, CUDA")
set_property(CACHE PLATFORM PROPERTY STRINGS "CPU;METAL;CUDA")


include(FetchContent)

# photon binary
file(
    GLOB SRCS
    "engine/*.cc" "engine/*.cpp" "engine/*.h"
    "schema/*.h" "schema/*.cc"
    "layers/*.h" "layers/*.cc"
)


if(PLATFORM STREQUAL "METAL")
    set(PLATFORM_MACRO -DPHOTON_METAL)
    add_library(photon STATIC ${SRCS})
else()
    set(PLATFORM_MACRO -DPHOTON_CPU)
    file(GLOB CPU_SRCS "cpu/*.cc" "cpu/*.h")
    add_library(photon STATIC ${SRCS} ${CPU_SRCS})
endif()

target_include_directories(photon PUBLIC . )


# schema
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece
    GIT_TAG        v0.2.0
)
FetchContent_MakeAvailable(sentencepiece)
target_link_libraries(photon sentencepiece-static)
target_include_directories(photon PUBLIC ${sentencepiece_SOURCE_DIR})


enable_testing()
add_subdirectory(unit_tests)
