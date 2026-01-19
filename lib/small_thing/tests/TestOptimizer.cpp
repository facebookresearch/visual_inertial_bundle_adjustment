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
using Mat11 = Eigen::Matrix<double, 1, 1>;
template <typename T>
using Ref = Eigen::Ref<T>;

TEST(TestOptimizer, Simple) {
  // points connected with springs
  static constexpr double springLen = 1.0;
  vector<Variable<Vec1>> pointVars = {{-2}, {-1}, {0}, {0.5}, {1.5}, {2.5}};

  Optimizer opt;
  for (size_t i = 0; i < pointVars.size() - 1; i++) {
    opt.addFactor(
        [=](const Vec1& x, const Vec1& y, Ref<Mat11>&& dx, Ref<Mat11>&& dy) -> Vec1 {
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

  // get hessian and indices
  auto [hessian, hessVarIndexAt] = opt.sparseElimMarginalInformation();

  // manual inverse of the hessian
  Eigen::MatrixXd invHessian =
      hessian.llt().solve(Eigen::MatrixXd::Identity(hessian.rows(), hessian.cols()));

  std::vector<int64_t> covVarIndices(pointVars.size());
  std::iota(covVarIndices.begin(), covVarIndices.end(), 0);

  auto covs = opt.computeCovariances(covVarIndices);
  ASSERT_EQ(covs.size(), covVarIndices.size());

  for (size_t i = 0; i < pointVars.size(); i++) {
    int hessIdx = hessVarIndexAt[i];
    ASSERT_NEAR(covs[hessIdx](0, 0), invHessian(i, i), 1e-7);
  }

  auto covAndOffsets = opt.computeJointCovariances({{1, 3, 4}});
  ASSERT_EQ(covAndOffsets.size(), 1);

  auto [cov, covOffs] = covAndOffsets[0];
  auto mProb = opt.computeMarginalProblem({1, 3, 4});
  ASSERT_EQ(covOffs.size(), mProb.offsets.size());

  for (size_t i = 0; i < covOffs.size(); i++) {
    ASSERT_EQ(i, covOffs[i]);
    ASSERT_EQ(i, mProb.offsets[i]);
  }

  Eigen::MatrixXd delta =
      (mProb.H * cov) - Eigen::MatrixXd::Identity(mProb.H.rows(), mProb.H.cols());
  ASSERT_NEAR(delta.norm(), 0.0, 1e-9);
}
