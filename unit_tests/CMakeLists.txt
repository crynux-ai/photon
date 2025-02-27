# cmake -DPLATFORM=METAL -DCMAKE_BUILD_TYPE=Debug .. && make

set(CMAKE_CXX_FLAGS_DEBUG "-g")

cmake_minimum_required(VERSION 3.12)

set(TESTS
    tokenizer_test
    feedforward_test
    rope_test
    attention_test
    transformer_test
)

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


if(PLATFORM STREQUAL "METAL")
    set(LINKS photon photon_metallib gtest_main)

    foreach(test IN LISTS TESTS)
        target_compile_options(${test} PRIVATE -x objective-c++)
        target_include_directories(${test} PRIVATE ${CMAKE_SOURCE_DIR}/metal)
    endforeach()
else()
    set(LINKS photon gtest_main)
endif()

    
foreach(test IN LISTS TESTS)
    target_compile_definitions(${test} PUBLIC ${PLATFORM_MACRO})
    target_link_libraries(${test} PRIVATE ${LINKS})
    add_test(NAME ${test} COMMAND ${test} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()

