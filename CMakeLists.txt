cmake_minimum_required(VERSION 3.11)
project("photon" CXX)

set(CMAKE_CXX_STANDARD 20)

option(BUILD_CPU "Build with CPU backend" ON)
option(BUILD_CUDA "Build with CUDA backend" OFF)
option(BUILD_METAL "Build with Metal backend" OFF)

include(FetchContent)

# photon binary
file(
    GLOB SRCS
    "engine/*.cc" "engine/*.cpp" "engine/*.h"
    "schema/*.h" "schema/*.cc"
    "layers/*.h" "layers/*.cc"
)


if(BUILD_METAL)
else()
    file(GLOB CPU_SRCS "cpu/*.cc" "cpu/*.h")
    add_library(photon STATIC ${SRCS} ${CPU_SRCS})
endif()

target_include_directories(photon PUBLIC . )



# layers
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
