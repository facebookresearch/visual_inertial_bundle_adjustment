# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

add_executable(
  TestCondensedFactor
  TestCondensedFactor.cpp
)
target_link_libraries(
  TestCondensedFactor ${BASPACHO_LIBRARY} small_thing gtest_main gmock
)
gtest_discover_tests(TestCondensedFactor)

add_executable(
  TestDynamicVars
  TestDynamicVars.cpp
)
target_link_libraries(
  TestDynamicVars ${BASPACHO_LIBRARY} small_thing gtest_main gmock
)
gtest_discover_tests(TestDynamicVars)

add_executable(
  TestOptimizer
  TestOptimizer.cpp
)
target_link_libraries(
  TestOptimizer ${BASPACHO_LIBRARY} small_thing gtest_main gmock
)
gtest_discover_tests(TestOptimizer)
