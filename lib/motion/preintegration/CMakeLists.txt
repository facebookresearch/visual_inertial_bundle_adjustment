# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

find_package(fmt REQUIRED)

add_library(preintegration
            MotionIntegral.cpp
            MotionIntegral.h
            ImuUtils.cpp
            ImuUtils.h
            ImuTypes.h
            CompensateJac.cpp
            CompensateJac.h
            PreIntegration.cpp
            PreIntegration.h
            RollingShutterData.cpp
            RollingShutterData.h)
target_include_directories(preintegration PUBLIC "../")
set_property(TARGET preintegration PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(preintegration fmt::fmt)

add_subdirectory("tests")
