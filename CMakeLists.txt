cmake_minimum_required(VERSION 3.11)

set(PLATFORM "CPU" CACHE STRING "Select an option: CPU, METAL, CUDA")
set_property(CACHE PLATFORM PROPERTY STRINGS "CPU;METAL;CUDA")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# photon binary
file(
    GLOB SRCS
    "engine/*.cc" "engine/*.cpp" "engine/*.h"
    "include/*.h"
    "schema/*.h" "schema/*.cc"
    "shared/*.h" "shared/*.cc"
)

if(PLATFORM STREQUAL "METAL")
    set(PLATFORM_MACRO -DPHOTON_METAL)
    project("photon" LANGUAGES C CXX OBJCXX)

    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

    # Build metal
    file(GLOB METAL_SHADER_FILES "metal/*.metal")
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/air_files"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/air_files"
    )

    set(METAL_AIR_FILES)
    foreach(shader_file IN LISTS METAL_SHADER_FILES)
        get_filename_component(shader_name ${shader_file} NAME_WE) # Get filename without extension
        set(_output "${CMAKE_CURRENT_BINARY_DIR}/air_files/${shader_name}.air")
        list(APPEND METAL_AIR_FILES "${_output}")
        add_custom_command(
            OUTPUT ${_output}
            DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/air_files"
            COMMAND xcrun -sdk macosx metal -c "${shader_file}" -o "${_output}"
        )
    endforeach()

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/photon.metallib
        DEPENDS ${METAL_AIR_FILES}
        COMMAND xcrun -sdk macosx metallib ${METAL_AIR_FILES} -o ${CMAKE_CURRENT_BINARY_DIR}/photon.metallib
        DEPENDS ${METAL_SHADER_FILES}
    )
    add_custom_target(metallib ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/photon.metallib)

    # Build objc lib
    file(GLOB METAL_SRC "metal/*.mm" "metal/*.h")
    add_library(photon_metallib STATIC ${METAL_SRC})
    target_sources(photon_metallib PRIVATE ${METAL_SRCS})
    target_include_directories(photon_metallib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(photon_metallib PRIVATE ${PLATFORM_MACRO})
    target_link_libraries(photon_metallib
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${QUARTZCORE_FRAMEWORK}
    )

    # Build photon
    add_library(photon STATIC ${SRCS})
    target_include_directories(photon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(photon PRIVATE ${PLATFORM_MACRO})
    target_link_libraries(photon photon_metallib)
    add_custom_command(
        TARGET photon POST_BUILD
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/photon.metallib
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_BINARY_DIR}/photon.metallib
            $<TARGET_FILE_DIR:photon>
        COMMENT "Copying photon.metallib to output directory"
    )
    add_custom_command(
        TARGET photon POST_BUILD
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/photon.metallib
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_BINARY_DIR}/photon.metallib
            ${CMAKE_CURRENT_BINARY_DIR}/unit_tests
        COMMENT "Copying photon.metallib to unit tests directory"
    )

    add_dependencies(photon metallib)
else()
    set(PLATFORM_MACRO -DPHOTON_CPU)
    project("photon" CXX)
    file(GLOB CPU_SRCS "cpu/*.cc" "cpu/*.h")
    add_library(photon STATIC ${SRCS} ${CPU_SRCS})
    target_compile_options(photon PRIVATE ${PLATFORM_MACRO})
endif()

target_include_directories(photon PUBLIC . )


# schema
include(FetchContent)
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
