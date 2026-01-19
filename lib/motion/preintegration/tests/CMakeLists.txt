# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

add_executable(
  TestMotionIntegral
  TestMotionIntegral.cpp
)
target_link_libraries(
  TestMotionIntegral preintegration gtest_main gmock
)
gtest_discover_tests(TestMotionIntegral)

add_executable(
  TestCompensateJac
  TestCompensateJac.cpp
)
target_link_libraries(
  TestCompensateJac preintegration gtest_main gmock
)
gtest_discover_tests(TestCompensateJac)

add_executable(
  TestPreIntegration
  TestPreIntegration.cpp
)
target_link_libraries(
  TestPreIntegration preintegration gtest_main gmock
)
gtest_discover_tests(TestPreIntegration)
