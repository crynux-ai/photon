cmake_minimum_required(VERSION 3.11)
project("photon" CXX)

set(CMAKE_CXX_STANDARD 17)


include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(layers)

enable_testing()
add_subdirectory(unit_tests)
