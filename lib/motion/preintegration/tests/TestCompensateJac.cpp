/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <preintegration/CompensateJac.h>
#include <preintegration/ImuUtils.h>
#include <random>

using namespace visual_inertial_ba;
using namespace visual_inertial_ba::preintegration;

constexpr double EPS = 1e-8;

// numeric jacobian, for comparison
static void getCompensatedImuMeasurementAndNumericJac(
    const ImuMeasurementModelParameters& imuParams,
    const SignalStatistics& uncompensatedGyroRadSec,
    const SignalStatistics& uncompensatedAccelMSec2,
    Vec3& compensatedGyroRadSec,
    Vec3& compensatedAccelMSec2,
    const ImuCalibrationJacobianIndices& jacInd,
    Ref<Mat6X> jacobian,
    Ref<Mat66> measJac) {
  ASSERT_EQ(jacobian.cols(), jacInd.getErrorStateSize());

  imuParams.getCompensatedImuMeasurement(
      uncompensatedGyroRadSec,
      uncompensatedAccelMSec2,
      compensatedGyroRadSec,
      compensatedAccelMSec2);

  // calib jacobian
  const size_t eStateSize = jacInd.getErrorStateSize();
  VecX delta(eStateSize);
  for (int i = 0; i < eStateSize; i++) {
    delta.setZero();
    delta[i] = EPS;
    ImuMeasurementModelParameters imuParamsP = imuParams;
    boxPlus(imuParamsP, jacInd, delta);
    Vec3 compensatedGyroRadSecP, compensatedAccelMSec2P;
    imuParamsP.getCompensatedImuMeasurement(
        uncompensatedGyroRadSec,
        uncompensatedAccelMSec2,
        compensatedGyroRadSecP,
        compensatedAccelMSec2P);

    jacobian.col(i) << (compensatedGyroRadSecP - compensatedGyroRadSec) / EPS,
        (compensatedAccelMSec2P - compensatedAccelMSec2) / EPS;
  }

  // meas jacobian
  for (int i = 0; i < 6; i++) {
    SignalStatistics uncompensatedGyroRadSecP = uncompensatedGyroRadSec;
    SignalStatistics uncompensatedAccelMSec2P = uncompensatedAccelMSec2;
    (i < 3 ? uncompensatedGyroRadSecP : uncompensatedAccelMSec2P).averageSignal[i % 3] += EPS;

    Vec3 compensatedGyroRadSecP, compensatedAccelMSec2P;
    imuParams.getCompensatedImuMeasurement(
        uncompensatedGyroRadSecP,
        uncompensatedAccelMSec2P,
        compensatedGyroRadSecP,
        compensatedAccelMSec2P);

    measJac.col(i) << (compensatedGyroRadSecP - compensatedGyroRadSec) / EPS,
        (compensatedAccelMSec2P - compensatedAccelMSec2) / EPS;
  }
}

template <typename T>
static T randVec(std::mt19937& g) {
  T retv;
  for (int i = 0; i < retv.size(); i++) {
    retv.data()[i] = std::normal_distribution<>(0, 1)(g);
  }
  return retv;
}

static VecX randVecX(int size, std::mt19937& g) {
  VecX retv(size);
  for (int i = 0; i < retv.size(); i++) {
    retv.data()[i] = std::normal_distribution<>(0, 1)(g);
  }
  return retv;
}

TEST(TestCompensateJac, CalibJac) {
  std::mt19937 g(42);
  ImuCalibrationOptions estOpts = ImuCalibrationOptions::allExceptTimeOffsets();
  ImuCalibrationJacobianIndices jacInd(estOpts);

  MatX jacMaxDelta = MatX::Zero(6, estOpts.errorStateSize());
  Mat66 measJacMaxDelta = Mat66::Zero();

  for (int q = 0; q < 10; q++) {
    ImuMeasurementModelParameters imuParams = factoryImuParams();

    // perturbed, randomly
    if (q > 1) {
      boxPlus(imuParams, jacInd, randVecX(estOpts.errorStateSize(), g) * 0.1);
    }

    for (int i = 0; i < 40; i++) {
      SignalStatistics uncompensatedGyroRadSec{
          .averageSignal = randVec<Vec3>(g),
          .rate = randVec<Vec3>(g),
      };
      SignalStatistics uncompensatedAccelMSec2{
          .averageSignal = randVec<Vec3>(g),
          .rate = randVec<Vec3>(g),
      };

      MatX numJac(6, estOpts.errorStateSize());
      Mat66 numMeasJac;
      Vec3 compensatedGyroRadSec, compensatedAccelMSec2;
      getCompensatedImuMeasurementAndNumericJac(
          imuParams,
          uncompensatedGyroRadSec,
          uncompensatedAccelMSec2,
          compensatedGyroRadSec,
          compensatedAccelMSec2,
          jacInd,
          numJac,
          numMeasJac);

      MatX anlJac(6, estOpts.errorStateSize());
      Mat66 anlMeasJac;
      Vec3 compensatedGyroRadSec_ALT, compensatedAccelMSec2_ALT;
      getCompensatedImuMeasurementAndJac(
          imuParams,
          uncompensatedGyroRadSec,
          uncompensatedAccelMSec2,
          compensatedGyroRadSec_ALT,
          compensatedAccelMSec2_ALT,
          jacInd,
          anlJac,
          anlMeasJac);

      EXPECT_LT((compensatedGyroRadSec - compensatedGyroRadSec_ALT).norm(), 1e-5);
      EXPECT_LT((compensatedAccelMSec2 - compensatedAccelMSec2_ALT).norm(), 1e-5);

      jacMaxDelta = (numJac - anlJac).cwiseAbs().cwiseMax(jacMaxDelta);
      measJacMaxDelta = (numMeasJac - anlMeasJac).cwiseAbs().cwiseMax(measJacMaxDelta);
    }
  }

  // print individual jacobians, useful for debugging
  std::cout << "Imu Calib Jacobian Delta:\n"
            << printableImuCalibJacDelta(
                   jacMaxDelta, jacInd, {{"gyro", 0, 3}, {"accel", 3, 3}}, false, 1.5e-6);
  std::cout << "Imu Measurement Jacobian Delta:\n"
            << printableImuMeasJacDelta(
                   measJacMaxDelta, {{"gyro", 0, 3}, {"accel", 3, 3}}, false, 5e-7);

  EXPECT_NEAR(jacMaxDelta.cwiseAbs().maxCoeff(), 0, 1.5e-6);
  EXPECT_NEAR(measJacMaxDelta.cwiseAbs().maxCoeff(), 0, 5e-7);

  // std::cout << "Imu Calib Jacobian Delta:\n" << measJacMaxDelta << std::endl;
}
