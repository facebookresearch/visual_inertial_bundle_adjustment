#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

cmake_minimum_required(VERSION 3.28)

# when HEAVY DEBUG is required:
# add_compile_options(-fsanitize=address)
# add_link_options(-fsanitize=address)
# set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE) # static
add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

# Notice: flags must be IDENTICAL to those in ProjectAria tools, to avoid ABI issues
add_compile_definitions(
  EIGEN_DONT_VECTORIZE
  EIGEN_DONT_ALIGN
)

# set the project name
project(VisualInertialBundleAdjustment CXX)
set(CMAKE_CXX_STANDARD 17)
set(WST_CXX_FLAGS -Wall -Wextra -pedantic) # -O1 -g -fsanitize=address -static-libsan)

message("* Build type: " ${CMAKE_BUILD_TYPE})

set(VIBA_BUILD_TESTS ON CACHE BOOL "If on (default), tests are build")

# some fancy colors for cmake messages
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Red "${Esc}[31m")
  set(Green "${Esc}[32m")
  set(Yellow "${Esc}[33m")
  set(Blue "${Esc}[34m")
  set(Magenta "${Esc}[35m")
  set(Cyan "${Esc}[36m")
  set(White "${Esc}[37m")
endif()

# use fetchcontent to download dependencies
include(FetchContent)
cmake_policy(SET CMP0077 NEW)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
  cmake_policy(SET CMP0167 OLD) # do not complain about FindBoost
  cmake_policy(SET CMP0169 OLD) # do not complain about FetchContent_Populate
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
  cmake_policy(SET CMP0135 NEW)
endif()

# prefer STATIC
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Main dependency: ProjectAria tools, used for VRS, MPS data loading and calibration types
message("${Cyan}=====================[ ProjectAria-Tools ]============================${ColourReset}")
add_subdirectory(deps/projectaria_tools)

if(false)
  # CSV Pasters
  message("${Cyan}====================[ Fast-CPP-CSV-Parser ]===========================${ColourReset}")
  include(ExternalProject)
  ExternalProject_Add(
    fast-cpp-csv-parser
    GIT_REPOSITORY https://github.com/ben-strasser/fast-cpp-csv-parser.git
    GIT_TAG origin/master
    SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/fast-cpp-csv-parser"
    # disable following, since it is not needed
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_DIR ""
    INSTALL_COMMAND ""
  )

  # CLI11
  message("${Cyan}===========================[ CLI11 ]==================================${ColourReset}")

  add_subdirectory(deps/CLI11)
  set(CLI11_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/CLI11/include")
  set(CLI11_LIBRARY cli11)
  link_directories("${CMAKE_CURRENT_BINARY_DIR}/deps/CLI11")

  message("* CLI11 Include Dir = " ${CLI11_INCLUDE_DIR})
  message("* CLI11 Libraries = " ${CLI11_LIBRARY})

  # Fmt
  message("${Cyan}==============================[ Fmt ]===================================${ColourReset}")
  add_subdirectory(deps/fmt)

  set(FMT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/fmt/include")
  set(FMT_LIBRARY fmt)
  set_property(TARGET fmt PROPERTY POSITION_INDEPENDENT_CODE ON)
  link_directories("${CMAKE_CURRENT_BINARY_DIR}/deps/fmt")
endif()

# eigen
message("${Cyan}=============================[ Eigen ]==================================${ColourReset}")
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
  EXCLUDE_FROM_ALL
)
FetchContent_GetProperties(eigen)

if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()

message("* Eigen Source = " ${eigen_SOURCE_DIR})
message("* Eigen Library Dir = " ${eigen_BINARY_DIR})
include_directories(${eigen_SOURCE_DIR})

# Sophus (only needed for BA problem in examples/benchmarks)
message("${Cyan}=============================[ Sophus ]=================================${ColourReset}")
FetchContent_Declare(
  sophus
  URL https://github.com/strasdat/Sophus/archive/refs/tags/1.24.6.zip
  EXCLUDE_FROM_ALL
)
FetchContent_GetProperties(sophus)

if(NOT sophus_POPULATED)
  FetchContent_Populate(sophus)
endif()

message("* Sophus Source = " ${sophus_SOURCE_DIR})
message("* Sophus Library Dir = " ${sophus_BINARY_DIR})
include_directories(${sophus_SOURCE_DIR})
add_compile_definitions(SOPHUS_USE_BASIC_LOGGING=1)

# dispenso for multithreading
message("${Cyan}============================[ dispenso ]================================${ColourReset}")
FetchContent_Declare(
  dispenso
  URL https://github.com/facebookincubator/dispenso/archive/refs/tags/v1.4.0.zip
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_MakeAvailable(dispenso)
unset(CMAKE_POSITION_INDEPENDENT_CODE)
include_directories(${dispenso_SOURCE_DIR} ${dispenso_SOURCE_DIR}/dispenso/third-party/moodycamel)
message("* Dispenso Source = " ${dispenso_SOURCE_DIR})
message("* Dispenso Library Dir = " ${dispenso_BINARY_DIR})

# gtest
message("${Cyan}=============================[ GTest ]==================================${ColourReset}")
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
message("* GTest Source = " ${gtest_SOURCE_DIR})
message("* GTest Library Dir = " ${gtest_BINARY_DIR})
message("* GMock Source = " ${gmock_SOURCE_DIR})
message("* GMock Library Dir = " ${gmock_BINARY_DIR})
include(GoogleTest)

# Sokol (header-only graphics library)
message("${Cyan}============================[ Sokol ]====================================${ColourReset}")
FetchContent_Declare(
  sokol
  URL https://github.com/floooh/sokol/archive/c66a1f0.zip
  EXCLUDE_FROM_ALL
)
FetchContent_GetProperties(sokol)
if(NOT sokol_POPULATED)
  FetchContent_Populate(sokol)

  # Patch sokol_imgui.h to support 32-bit indices dynamically
  set(SOKOL_IMGUI_HEADER "${sokol_SOURCE_DIR}/util/sokol_imgui.h")
  file(READ "${SOKOL_IMGUI_HEADER}" SOKOL_IMGUI_CONTENT)

  # Find the line with pip_desc.index_type = SG_INDEXTYPE_UINT16; and replace it
  string(REPLACE
    "pip_desc.index_type = SG_INDEXTYPE_UINT16;"
    "pip_desc.index_type = sizeof(ImDrawIdx) == 2 ? SG_INDEXTYPE_UINT16 : SG_INDEXTYPE_UINT32;"
    SOKOL_IMGUI_CONTENT "${SOKOL_IMGUI_CONTENT}")

  file(WRITE "${SOKOL_IMGUI_HEADER}" "${SOKOL_IMGUI_CONTENT}")
  message("* Patched sokol_imgui.h for 32-bit index support")
endif()
message("* Sokol Source = " ${sokol_SOURCE_DIR})
include_directories(${sokol_SOURCE_DIR})

# ImGui
message("${Cyan}============================[ ImGui ]====================================${ColourReset}")
FetchContent_Declare(
  imgui
  URL https://github.com/ocornut/imgui/archive/refs/tags/v1.92.5-docking.zip
  EXCLUDE_FROM_ALL
)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
  FetchContent_Populate(imgui)

  # Patch imconfig.h to enable 32-bit indices
  set(IMGUI_CONFIG_HEADER "${imgui_SOURCE_DIR}/imconfig.h")
  file(READ "${IMGUI_CONFIG_HEADER}" IMGUI_CONFIG_CONTENT)

  # Uncomment the ImDrawIdx line
  string(REPLACE
    "//#define ImDrawIdx unsigned int"
    "#define ImDrawIdx unsigned int"
    IMGUI_CONFIG_CONTENT "${IMGUI_CONFIG_CONTENT}")

  file(WRITE "${IMGUI_CONFIG_HEADER}" "${IMGUI_CONFIG_CONTENT}")
  message("* Patched imconfig.h for 32-bit index support")
endif()
message("* ImGui Source = " ${imgui_SOURCE_DIR})

# ImPlot
message("${Cyan}============================[ ImPlot ]===================================${ColourReset}")
FetchContent_Declare(
  implot
  URL https://github.com/epezent/implot/archive/refs/tags/v0.17.zip
  EXCLUDE_FROM_ALL
)
FetchContent_GetProperties(implot)
if(NOT implot_POPULATED)
  FetchContent_Populate(implot)
endif()
message("* ImPlot Source = " ${implot_SOURCE_DIR})


# Baspacho (Force off: CUDA/tests/examples)
message("${Cyan}============================[ BaSpaCho ]=================================${ColourReset}")
set(BASPACHO_USE_CUBLAS OFF CACHE BOOL "If on, CUDA support is enabled" FORCE)
set(BASPACHO_BUILD_TESTS OFF CACHE BOOL "If on (default), tests are build" FORCE)
set(BASPACHO_BUILD_EXAMPLES OFF CACHE BOOL "If on (default), exampels/benchmarks are build" FORCE)

add_subdirectory("deps/baspacho")
include_directories("${PROJECT_SOURCE_DIR}/deps/baspacho")

link_directories("${CMAKE_CURRENT_BINARY_DIR}/deps/baspacho/baspacho/baspacho")

if(BUILD_SHARED_LIBS)
  set(BASPACHO_LIBRARY BaSpaCho)
else()
  set(BASPACHO_LIBRARY BaSpaCho_static)
endif()

# ##########################################################################################################
# add root to include path
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/deps")

# enable tests, unless explicitly disabled
if(VIBA_BUILD_TESTS)
  enable_testing()
endif()

include_directories("${CMAKE_BINARY_DIR}/_deps") # for the CSV parser, be able to include <fast-cpp-csv-parser/csv.h>

add_subdirectory("lib")

add_subdirectory("interfaces")

add_subdirectory("viba")
