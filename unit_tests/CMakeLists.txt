# cmake -DPLATFORM=METAL -DCMAKE_BUILD_TYPE=Debug .. && make

set(CMAKE_CXX_FLAGS_DEBUG "-g")

cmake_minimum_required(VERSION 3.12)

add_executable(tokenizer_test "layers/tokenizer_test.cc")
add_executable(feedforward_test "layers/feedforward_test.cc")
add_executable(rope_test "layers/rope_test.cc")
add_executable(attention_test "layers/attention_test.cc")
add_executable(transformer_test "layers/transformer_test.cc")

# gtest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

target_compile_definitions(tokenizer_test PUBLIC ${PLATFORM_MACRO})
target_compile_definitions(feedforward_test PUBLIC ${PLATFORM_MACRO})
target_compile_definitions(rope_test PUBLIC ${PLATFORM_MACRO})
target_compile_definitions(attention_test PUBLIC ${PLATFORM_MACRO})
target_compile_definitions(transformer_test PUBLIC ${PLATFORM_MACRO})

target_link_libraries(tokenizer_test PRIVATE photon gtest_main)
target_link_libraries(feedforward_test PRIVATE photon gtest_main)
target_link_libraries(rope_test PRIVATE photon gtest_main)
target_link_libraries(attention_test PRIVATE photon gtest_main)
target_link_libraries(transformer_test PRIVATE photon gtest_main)

# Register the test with CTest
find_package(GTest REQUIRED)
add_test(NAME TokenzerTest COMMAND tokenizer_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME FeedforwardTest COMMAND feedforward_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME RopeTest COMMAND rope_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME AttentionTest COMMAND attention_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME TransformerTest COMMAND transformer_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
