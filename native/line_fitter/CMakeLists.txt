cmake_minimum_required(VERSION 3.21)
project(socratic_line_fitter LANGUAGES C)

option(SOCRATIC_BUILD_PRED_FIXUP "Build the TIFF grid command-line tool" OFF)
option(SOCRATIC_BUILD_RESEARCH_TESTS "Build slab/viz negative-result tests" OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(socratic_line_fitter STATIC
    src/common/arena.c
    src/common/except.c
    src/common/union_find.c
    src/fixup/bridge_scan.c
    src/fixup/join_paint.c
    src/fixup/join_tracks.c
    src/fixup/slice_match.c
    src/fixup/slice_trace.c
)
target_include_directories(socratic_line_fitter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(NOT MSVC)
    target_link_libraries(socratic_line_fitter PUBLIC m)
endif()
enable_testing()
add_executable(line_fitter_selftest tests/line_fitter_selftest.c)
target_link_libraries(line_fitter_selftest PRIVATE socratic_line_fitter)
add_test(NAME line_fitter_selftest COMMAND line_fitter_selftest)

if(SOCRATIC_BUILD_RESEARCH_TESTS OR SOCRATIC_BUILD_PRED_FIXUP)
    find_package(OpenMP REQUIRED COMPONENTS C)
    add_library(socratic_line_fitter_research STATIC
        src/common/ves_png.c
        src/fixup/fixup_viz.c
        src/fixup/slab_split.c
    )
    target_include_directories(socratic_line_fitter_research
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )
    target_link_libraries(socratic_line_fitter_research
        PUBLIC socratic_line_fitter OpenMP::OpenMP_C
    )
endif()

if(SOCRATIC_BUILD_RESEARCH_TESTS)
    add_executable(line_fitter_research_selftest tests/fixup_test.c)
    target_link_libraries(line_fitter_research_selftest
        PRIVATE socratic_line_fitter_research
    )
    add_test(NAME line_fitter_research_selftest COMMAND line_fitter_research_selftest)
endif()

if(SOCRATIC_BUILD_PRED_FIXUP)
    find_package(TIFF REQUIRED)
    add_executable(pred_fixup
        src/common/tiff_io.c
        src/common/ves_platform.c
        src/tools/pred_fixup.c
    )
    target_link_libraries(pred_fixup
        PRIVATE socratic_line_fitter_research TIFF::TIFF
    )
endif()
