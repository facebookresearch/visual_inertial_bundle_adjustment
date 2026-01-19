# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

find_package(fmt REQUIRED)

add_library(viba_common
    Constants.h
    Enumerate.h
    FindOrDie.h
    Format.h
    Histogram.h
    Report.h
    Settings.h
    StatsValueContainer.h
    Utf8Symbols.h
    Histogram.cpp
    Report.cpp
    Settings.cpp
    StatsValueContainer.cpp
)
set_property(TARGET viba_common PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(viba_common small_thing CLI11::CLI11 fmt::fmt nlohmann_json::nlohmann_json vrs_logging)
target_include_directories(viba_common PUBLIC "../..")
