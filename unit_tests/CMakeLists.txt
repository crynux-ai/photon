cmake_minimum_required(VERSION 3.12)

file(GLOB SRCS "**/*.cc" "**/*.h")
add_executable(tests ${SRCS})

# gtest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

target_link_libraries(
    tests PRIVATE
    layers
    gtest_main
)

# Register the test with CTest
add_test(
    NAME RunTests
    COMMAND tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
