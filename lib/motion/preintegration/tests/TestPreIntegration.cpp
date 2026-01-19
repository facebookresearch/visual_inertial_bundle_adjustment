/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <preintegration/CompensateJac.h>
#include <preintegration/ImuUtils.h>
#include <preintegration/PreIntegration.h>
#include <random>

using namespace visual_inertial_ba;
using namespace visual_inertial_ba::preintegration;

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

static std::vector<ImuMeasurement>
genImuMeasurements(std::mt19937& g, int64_t timeStartUs, int64_t timeEndUs) {
  std::vector<ImuMeasurement> retv;

  const int64_t kTimeDeltaUs = 1000; // 1ms
  for (int64_t t = timeStartUs; t < timeEndUs; t += kTimeDeltaUs) {
    retv.push_back(newImuMeasurement(
        randVec<Vec3>(g) / std::sqrt(3.0) * 9.81 * 2,
        randVec<Vec3>(g) / std::sqrt(3.0) * M_PI,
        t * 1000));
  }

  return retv;
}

static void randomizeImuMeasurements(
    std::mt19937& g,
    const std::vector<ImuMeasurement>& meas,
    const ImuNoiseModelParameters& noiseModel,
    std::vector<ImuMeasurement>& measOut) {
  const Vec3 accelSampleStdDevMSec2 = noiseModel.accelSampleVarianceM2Sec4.cwiseSqrt();
  const Vec3 gyroSampleStdDevRadSec = noiseModel.gyroSampleVarianceRad2Sec2.cwiseSqrt();
  measOut.resize(meas.size());
  for (int i = 0; i < meas.size(); i++) {
    measOut[i] = meas[i];
    measOut[i].accelMSec2 += accelSampleStdDevMSec2.asDiagonal() * randVec<Vec3>(g);
    measOut[i].gyroRadSec += gyroSampleStdDevRadSec.asDiagonal() * randVec<Vec3>(g);
  }
}

static PreIntegration computeNumJac(
    const ImuCalibrationJacobianIndices& jacInd,
    const std::vector<ImuMeasurement>& meas,
    const ImuMeasurementModelParameters& measModel,
    const ImuNoiseModelParameters& noiseModel,
    int64_t timeStartUs,
    int64_t timeEndUs) {
  const size_t eStateSize = jacInd.getErrorStateSize();

  VecX delta(eStateSize);
  Mat9X J(9, eStateSize);

  auto rvp0 = integrateMeasurements(meas, measModel, timeStartUs, timeEndUs);
  for (int i = 0; i < eStateSize; i++) {
    const double EPS = i == jacInd.referenceImuTimeOffsetIdx() ? 1e-7
        : i == jacInd.gyroAccelTimeOffsetIdx()                 ? 3e-9
                                                               : 1e-6;

    ImuMeasurementModelParameters pertModelMinus = measModel, pertModelPlus = measModel;
    delta.setZero();
    delta[i] = EPS;
    preintegration::boxPlus(pertModelPlus, jacInd, delta);
    delta[i] = -EPS;
    preintegration::boxPlus(pertModelMinus, jacInd, delta);

    auto pertRvpMinus = integrateMeasurements(meas, pertModelMinus, timeStartUs, timeEndUs);
    auto pertRvpPlus = integrateMeasurements(meas, pertModelPlus, timeStartUs, timeEndUs);
    J.col(i) = (boxMinus(pertRvpPlus, rvp0) - boxMinus(pertRvpMinus, rvp0)) / (2.0 * EPS);
  }

  return {.rvp = rvp0, .J = J};
}

template <typename M>
[[maybe_unused]] static void capNorm(M& m, double maxNorm) {
  if (m.norm() > maxNorm) {
    m *= maxNorm / m.norm();
  }
}

TEST(TestPreIntegration, PreInt) {
  std::mt19937 g(43);
  ImuCalibrationOptions estOpts(true);
  ImuCalibrationJacobianIndices jacInd(estOpts);

  ImuNoiseModelParameters noiseModel; // defaults

  MatX jacMaxDelta(9, estOpts.errorStateSize());
  for (int q = 0; q < 250; q++) {
    ImuMeasurementModelParameters imuParams = factoryImuParams();

    // perturbed, randomly
    if (q > 2) {
      boxPlus(imuParams, jacInd, randVecX(estOpts.errorStateSize(), g) * 0.05);
    }
    if (q % 2 == 0) {
      // for accel/gyro aligned, to test corner case for jacobian
      imuParams.dtReferenceAccelSec = imuParams.dtReferenceGyroSec;
    }

    int64_t timeStartUs = 850'000;
    int64_t timeEndUs = 1'150'000;

    for (int w = 0; w < 5; w++) {
      auto meas = genImuMeasurements(g, 0, 2'000'000);
      auto myPreintRes = preintegration::computePreIntegration(
          jacInd, meas, imuParams, noiseModel, timeStartUs, timeEndUs);
      auto myPreintNumJacRes =
          computeNumJac(jacInd, meas, imuParams, noiseModel, timeStartUs, timeEndUs);

      MatX relDelta = (myPreintRes.J - myPreintNumJacRes.J)
                          .cwiseQuotient(myPreintNumJacRes.J.cwiseAbs().cwiseMax(1.0));
      jacMaxDelta = relDelta.cwiseAbs().cwiseMax(jacMaxDelta);
    }
  }

  std::cout << "Imu Calib Jacobian Delta:\n"
            << printableImuCalibJacDelta(
                   jacMaxDelta, jacInd, {{"rot", 0, 3}, {"vel", 3, 3}, {"pos", 6, 3}}, false, 1e-4);
  const int eStateSizeExceptTimeOffsets = estOpts.errorStateSize() -
      (jacInd.referenceImuTimeOffsetIdx() >= 0) - (jacInd.gyroAccelTimeOffsetIdx() >= 0);
  EXPECT_NEAR(jacMaxDelta.leftCols(eStateSizeExceptTimeOffsets).cwiseAbs().maxCoeff(), 0, 1e-6);
  EXPECT_NEAR(jacMaxDelta.col(jacInd.referenceImuTimeOffsetIdx()).cwiseAbs().maxCoeff(), 0, 1e-6);
  EXPECT_NEAR(jacMaxDelta.col(jacInd.gyroAccelTimeOffsetIdx()).cwiseAbs().maxCoeff(), 0, 1e-4);
}

TEST(TestPreIntegration, Covariance) {
  ImuCalibrationOptions estOpts(true);
  ImuCalibrationJacobianIndices jacInd(estOpts);

  ImuNoiseModelParameters noiseModel; // defaults

  MatX jacMaxDelta(9, estOpts.errorStateSize());
  for (int q = 0; q < 20; q++) {
    int seed = 39 + q;
    std::mt19937 g(seed);
    ImuMeasurementModelParameters imuParams = factoryImuParams();

    // perturbed, randomly
    if (q > 1) {
      boxPlus(imuParams, jacInd, randVecX(estOpts.errorStateSize(), g) * 0.005);
    }

    int64_t timeStartUs = 50'000;
    int64_t timeEndUs = 150'000;

    auto meas = genImuMeasurements(g, 0, 200'000);
    std::vector<ImuMeasurement> randMeas;

    auto myPreintRes = preintegration::computePreIntegration(
        jacInd, meas, imuParams, noiseModel, timeStartUs, timeEndUs);

    const Mat99 noiseCol = myPreintRes.rvpCov.llt().matrixL();
    const Mat99 whiteNoise = noiseCol.triangularView<Eigen::Lower>().solve(Mat99::Identity());

    int64_t numSamples = 250'000;
    int64_t addedSamples = 0;
    Mat99 sampleWhiteCov = Mat99::Zero();
    for (int i = 0; i < numSamples; i++) {
      randomizeImuMeasurements(g, meas, noiseModel, randMeas);
      auto rvpRand = integrateMeasurements(randMeas, imuParams, timeStartUs, timeEndUs);

      Vec9 wnSample = whiteNoise * boxMinus(rvpRand, myPreintRes.rvp);
      if (wnSample.squaredNorm() > 9 + 4 * std::sqrt(2 * 9)) { // 4-sigma (approx Chi2 test)
        continue;
      }
      sampleWhiteCov += wnSample * wnSample.transpose();
      addedSamples++;
    }
    sampleWhiteCov /= addedSamples;

    auto svd = sampleWhiteCov.jacobiSvd();
    std::cout << "g=" << seed << ", nsamples: " << addedSamples << " / " << numSamples
              << ", SVs: \n"
              << svd.singularValues().transpose() << "\n";

    EXPECT_NEAR(svd.singularValues()[0], 1, 0.04);
    EXPECT_NEAR(svd.singularValues()[8], 1, 0.04);
  }
}
