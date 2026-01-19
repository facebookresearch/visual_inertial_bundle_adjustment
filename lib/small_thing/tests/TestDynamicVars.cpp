/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <baspacho/testing/TestingUtils.h>
#include <gtest/gtest.h>
#include <small_thing/Optimizer.h>
#include <chrono>
#include <iomanip>

using namespace small_thing;
using namespace BaSpaCho::testing_utils;
using namespace std;
using Vec1 = Eigen::Vector<double, 1>;
using VecX = Eigen::Vector<double, Eigen::Dynamic>;
using Mat1X = Eigen::Matrix<double, 1, Eigen::Dynamic>;
using Mat11 = Eigen::Matrix<double, 1, Eigen::Dynamic>;
template <typename T>
using RefK = Eigen::Ref<const T>;
template <typename T>
using Ref = Eigen::Ref<T>;

TEST(TestOptimizer, Simple) {
  // points connected with springs
  static constexpr double springLen = 1.0;
  vector<Variable<VecX>> pointVars = {Vec1{-2}, Vec1{-1}, Vec1{0}, Vec1{0.5}, Vec1{1.5}, Vec1{2.5}};

  Optimizer opt;
  for (size_t i = 0; i < pointVars.size() - 1; i++) {
    opt.addFactor(
        [=](const VecX& x, const VecX& y, Ref<Mat1X>&& dx, Ref<Mat1X>&& dy) -> Vec1 {
          if (!isNull(dx)) {
            dx(0, 0) = -1;
          }
          if (!isNull(dy)) {
            dy(0, 0) = 1;
          }
          return Vec1(y[0] - x[0] - springLen);
        },
        pointVars[i],
        pointVars[i + 1]);
  }

  ASSERT_TRUE(opt.verifyJacobians());

  opt.optimize();

  for (size_t i = 0; i < pointVars.size() - 1; i++) {
    ASSERT_NEAR(pointVars[i + 1].value[0] - pointVars[i].value[0], springLen, 1e-8);
  }
}

// here the factor assumes compile-time-known sizes. This should work thanks to the
// Eigen::Ref magic, as long as the size is the right one at runtime
TEST(TestOptimizer, ConstInFactor) {
  // points connected with springs
  static constexpr double springLen = 1.0;
  vector<Variable<VecX>> pointVars = {Vec1{-2}, Vec1{-1}, Vec1{0}, Vec1{0.5}, Vec1{1.5}, Vec1{2.5}};

  Optimizer opt;
  for (size_t i = 0; i < pointVars.size() - 1; i++) {
    opt.addFactor(
        [=](const RefK<Vec1>& x, const RefK<Vec1>& y, Ref<Mat11>&& dx, Ref<Mat11>&& dy) -> Vec1 {
          if (!isNull(dx)) {
            dx(0, 0) = -1;
          }
          if (!isNull(dy)) {
            dy(0, 0) = 1;
          }
          return Vec1(y[0] - x[0] - springLen);
        },
        pointVars[i],
        pointVars[i + 1]);
  }

  ASSERT_TRUE(opt.verifyJacobians());

  opt.optimize();

  for (size_t i = 0; i < pointVars.size() - 1; i++) {
    ASSERT_NEAR(pointVars[i + 1].value[0] - pointVars[i].value[0], springLen, 1e-8);
  }
}
