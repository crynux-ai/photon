# cmake -DCMAKE_BUILD_TYPE=Debug .. && make

cmake_minimum_required(VERSION 3.12)

file(GLOB SRCS "**/*.cc" "**/*.h")
add_executable(tokenizer_test "layers/tokenizer_test.cc")
add_executable(feedforward_test "layers/feedforward_test.cc")
add_executable(rope_test "layers/rope_test.cc")
add_executable(attention_test "layers/attention_test.cc")

# gtest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

target_link_libraries(tokenizer_test PRIVATE layers gtest_main)
target_link_libraries(feedforward_test PRIVATE layers gtest_main)
target_link_libraries(rope_test PRIVATE layers gtest_main)
target_link_libraries(attention_test PRIVATE layers gtest_main)

# Register the test with CTest
add_test(NAME TokenizerTest COMMAND tokenizer_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME FeedforwardTest COMMAND feedforward_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME RopeTest COMMAND rope_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME AttentionTest COMMAND attention_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
