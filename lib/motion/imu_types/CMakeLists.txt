# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

cmake_policy(SET CMP0076 NEW) # Allow relative path in target_include_directories and INTERFACE

add_library(imu_types
  ImuDataReader.cpp
  ImuDataWriter.cpp
  ImuCalibrationJacobianIndices.h
  ImuDataFormat.h
  ImuDataWriter.h
  ImuMeasurementModelParameters.h
  SignalStatistics.h
  ImuCalibrationOptions.h
  ImuDataReader.h
  ImuMeasurement.h
  ImuNoiseModelParameters.h
)
add_dependencies(imu_types fast-cpp-csv-parser)
target_include_directories(imu_types PUBLIC "../")
set_property(TARGET imu_types PROPERTY POSITION_INDEPENDENT_CODE ON)
