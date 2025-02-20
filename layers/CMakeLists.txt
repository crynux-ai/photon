cmake_minimum_required(VERSION 3.11)
set(CMAKE_CXX_FLAGS_DEBUG "-g")


file(GLOB SRCS "*.cc" "*.h")
add_library(layers STATIC ${SRCS})


include(FetchContent)

FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece
    GIT_TAG        v0.2.0
)
FetchContent_MakeAvailable(sentencepiece)

target_link_libraries(layers sentencepiece-static)
target_include_directories(layers PUBLIC ${sentencepiece_SOURCE_DIR})
