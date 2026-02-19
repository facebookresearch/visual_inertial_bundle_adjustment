# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

add_executable(
  TestInertialFactor
  TestInertialFactor.cpp
)
target_link_libraries(
  TestInertialFactor ark_problem preintegration gtest_main gmock
)
gtest_discover_tests(TestInertialFactor)
