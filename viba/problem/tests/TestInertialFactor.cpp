/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <imu_model/ImuCalibParam.h>
#include <preintegration/CompensateJac.h>
#include <preintegration/ImuUtils.h>
#include <preintegration/PreIntegration.h>
#include <small_thing/Common.h>
#include <small_thing/Variable.h>
#include <viba/problem/Types.h>
#include <random>

using namespace visual_inertial_ba;
using namespace visual_inertial_ba::preintegration;
using small_thing::isNull;
using small_thing::NullRef;

using GravityData = small_thing::S2;

// ---------------------------------------------------------------------------
// Local replica of the InertialFactor residual + Jacobian computation.
// This must be kept in sync with InertialFactor.cpp.
// ---------------------------------------------------------------------------
static Vec9 evaluateInertialFactor(
    const PreIntegration& preint,
    const ImuCalibParam& calib,
    const SE3& prev_T_imu_world,
    const Vec3& prev_vel_world,
    const SE3& next_T_imu_world,
    const Vec3& next_vel_world,
    const GravityData& gravityWorld,
    Eigen::Ref<Mat9X> calib_Jacobian,
    Eigen::Ref<Mat96> prev_T_imu_world_Jacobian,
    Eigen::Ref<Mat93> prev_vel_world_Jacobian,
    Eigen::Ref<Mat96> next_T_imu_world_Jacobian,
    Eigen::Ref<Mat93> next_vel_world_Jacobian,
    Eigen::Ref<Mat92> gravityWorld_Jacobian) {
  const int preintErrorSize = calib.jacInd->getErrorStateSize();
  const double deltaTimeSec = preint.rvp.dtSec;

  // compute correction as consequence of variation in calibration
  Eigen::Map<VecX> deltaCalib(
      (double*)alloca(sizeof(double) * preintErrorSize), preintErrorSize);
  calib.boxMinus(preint.calibEvalPoint, deltaCalib);
  const Vec9 preintCorrection = preint.J * deltaCalib;

  // rotation error
  const SO3 R_rotCorrection = SO3::exp(-preintCorrection.head<3>());
  const SO3 corrected_R_ImuNext_ImuPrev =
      preint.rvp.R.inverse() * R_rotCorrection;
  const SO3 R_rotErr = corrected_R_ImuNext_ImuPrev * prev_T_imu_world.so3() *
      next_T_imu_world.so3().inverse();
  const Vec3 logRotErr = -R_rotErr.log();

  // velocity error
  const Vec3 deltaVelocity_world_est =
      next_vel_world - prev_vel_world - gravityWorld.vec * deltaTimeSec;
  const Vec3 deltaVelocity_prevImu_est =
      prev_T_imu_world.so3() * deltaVelocity_world_est;
  const Vec3 velErr = preint.rvp.dV - deltaVelocity_prevImu_est +
      preintCorrection.segment<3>(3);

  // position error
  const SO3 R_prevImu_nextImu =
      prev_T_imu_world.so3() * next_T_imu_world.so3().inverse();
  const Vec3 deltaPosition_prevImu_est = prev_T_imu_world.translation() -
      R_prevImu_nextImu * next_T_imu_world.translation() -
      prev_T_imu_world.so3() *
          (prev_vel_world * deltaTimeSec +
           gravityWorld.vec * (0.5 * deltaTimeSec * deltaTimeSec));
  const Vec3 posErr = preint.rvp.dP - deltaPosition_prevImu_est +
      preintCorrection.segment<3>(6);

  const Mat33 dLogRotErr_dLeftRotError = SO3::leftJacobianInverse(-logRotErr);

  // clang-format off
  if (!isNull(prev_T_imu_world_Jacobian)) {
    prev_T_imu_world_Jacobian <<
        Mat33::Zero(),      -dLogRotErr_dLeftRotError * corrected_R_ImuNext_ImuPrev.Adj(),
        Mat33::Zero(),      -SO3::hat(-deltaVelocity_prevImu_est),
        -Mat33::Identity(), -SO3::hat(-deltaPosition_prevImu_est);
  }
  if (!isNull(prev_vel_world_Jacobian)) {
    const Mat33 dPrevImuPt_dWorldPt = prev_T_imu_world.so3().matrix();
    prev_vel_world_Jacobian <<
        Mat33::Zero(),
        dPrevImuPt_dWorldPt,
        dPrevImuPt_dWorldPt * deltaTimeSec;
  }
  if (!isNull(next_T_imu_world_Jacobian)) {
    next_T_imu_world_Jacobian <<
        Mat33::Zero(), dLogRotErr_dLeftRotError * R_rotErr.Adj(),
        Mat36::Zero(),
        R_prevImu_nextImu.matrix(), Mat33::Zero();
  }
  if (!isNull(next_vel_world_Jacobian)) {
    next_vel_world_Jacobian <<
        Mat33::Zero(),
        -prev_T_imu_world.so3().matrix(),
        Mat33::Zero();
  }
  if (!isNull(gravityWorld_Jacobian)) {
    Mat32 dVel_dG = deltaTimeSec * prev_T_imu_world.so3().matrix() *
        small_thing::S2::ortho(gravityWorld.vec).transpose();
    gravityWorld_Jacobian <<
        Mat32::Zero(),
        dVel_dG,
        dVel_dG * (0.5 * deltaTimeSec);
  }
  // clang-format on
  if (!isNull(calib_Jacobian)) {
    const Mat33 dLogRotErr_dResidualCorrection = dLogRotErr_dLeftRotError *
        corrected_R_ImuNext_ImuPrev.Adj() *
        SO3::leftJacobianInverse(-preintCorrection.head<3>());
    const int calibErrorSize = calib.estOpts->errorStateSize();
    assert(calib_Jacobian.cols() == calibErrorSize);

    const auto& calibJac = preint.J;
    assert(calibErrorSize == calibJac.cols());

    calib_Jacobian << dLogRotErr_dResidualCorrection * calibJac.topRows<3>(),
        calibJac.bottomRows<6>();
  }

  Vec9 rotVelPosError;
  rotVelPosError << logRotErr, velErr, posErr;
  return rotVelPosError;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
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
  const int64_t kTimeDeltaUs = 1000;
  for (int64_t t = timeStartUs; t < timeEndUs; t += kTimeDeltaUs) {
    retv.push_back(newImuMeasurement(
        randVec<Vec3>(g) / std::sqrt(3.0) * 9.81 * 2,
        randVec<Vec3>(g) / std::sqrt(3.0) * M_PI,
        t * 1000));
  }
  return retv;
}

template <typename M>
static void capNorm(M& m, double maxNorm) {
  if (m.norm() > maxNorm) {
    m *= maxNorm / m.norm();
  }
}

/// Derive next state from preintegration so that the residual is exactly zero.
static std::pair<SE3, Vec3> deriveNextState(
    const PreIntegration& preint,
    const SE3& prev_T_imu_world,
    const Vec3& prev_vel_world,
    const Vec3& gravity_world) {
  const double dt = preint.rvp.dtSec;
  const SO3& R_prev = prev_T_imu_world.so3();

  // R_next = R_preint^{-1} * R_prev
  const SO3 R_next = preint.rvp.R.inverse() * R_prev;

  // v_next = v_prev + g*dt + R_prev^{-1} * d_v
  const Vec3 next_vel_world =
      prev_vel_world + gravity_world * dt + R_prev.inverse() * preint.rvp.dV;

  // R_preint * t_next = t_prev - R_prev*(v_prev*dt + g*dt^2/2) - d_p
  const Vec3 rhs = prev_T_imu_world.translation() -
      R_prev * (prev_vel_world * dt + gravity_world * (0.5 * dt * dt)) -
      preint.rvp.dP;
  const Vec3 t_next = preint.rvp.R.inverse() * rhs;

  return {SE3(R_next, t_next), next_vel_world};
}

/// Evaluate factor with no Jacobians.
static Vec9 evalResidualOnly(
    const PreIntegration& preint,
    const ImuCalibParam& calib,
    const SE3& prev_T,
    const Vec3& prev_vel,
    const SE3& next_T,
    const Vec3& next_vel,
    const GravityData& grav) {
  return evaluateInertialFactor(
      preint, calib, prev_T, prev_vel, next_T, next_vel, grav,
      NullRef(), NullRef(), NullRef(), NullRef(), NullRef(), NullRef());
}

// ---------------------------------------------------------------------------
// TEST: Jacobians
// ---------------------------------------------------------------------------
static void runJacobianTest(bool perturbNextState) {
  const char* label = perturbNextState ? "nonzero-residual" : "zero-residual";
  std::mt19937 g(42);

  ImuCalibrationOptions estOpts(true); // all options
  ImuCalibrationJacobianIndices jacInd(estOpts);
  ImuNoiseModelParameters noiseModel;
  const int eStateSize = estOpts.errorStateSize();

  double maxErrPrevPose = 0, maxErrPrevVel = 0;
  double maxErrNextPose = 0, maxErrNextVel = 0;
  double maxErrGravity = 0;
  double maxErrCalib = 0, maxErrCalibTime = 0;

  const int64_t timeStartUs = 850'000;
  const int64_t timeEndUs = 1'150'000;

  for (int trial = 0; trial < 50; trial++) {
    ImuMeasurementModelParameters imuParams = factoryImuParams();
    if (trial > 2) {
      boxPlus(imuParams, jacInd, randVecX(eStateSize, g) * 0.05);
    }
    if (trial % 2 == 1) {
      imuParams.dtReferenceAccelSec = imuParams.dtReferenceGyroSec;
    }

    auto meas = genImuMeasurements(g, 0, 2'000'000);
    auto preint = computePreIntegration(
        jacInd, meas, imuParams, noiseModel, timeStartUs, timeEndUs);

    // Random previous state
    const SO3 prevRot = SO3::exp(randVec<Vec3>(g));
    const Vec3 prevTrans = randVec<Vec3>(g) * 2.0;
    const SE3 prevT(prevRot, prevTrans);
    const Vec3 prevVel = randVec<Vec3>(g);

    // Random gravity
    Vec3 gDir = randVec<Vec3>(g);
    capNorm(gDir, 1.0);
    const Vec3 gravityVec = gDir.normalized() * 9.81;
    const GravityData gravity{.radius = gravityVec.norm(), .vec = gravityVec};

    // Derive next state
    auto [nextT, nextVel] =
        deriveNextState(preint, prevT, prevVel, gravityVec);

    if (perturbNextState) {
      Vec3 rotPert = randVec<Vec3>(g).normalized() * 0.087;
      Vec3 transPert = randVec<Vec3>(g).normalized() * 0.03;
      Vec6 se3Pert;
      se3Pert << transPert, rotPert; // Sophus convention: [trans, rot]
      nextT = SE3::exp(se3Pert) * nextT;
      nextVel += randVec<Vec3>(g).normalized() * 0.03;
    }

    // Evaluate with analytic Jacobians
    ImuCalibParam calib;
    calib.modelParams = imuParams;
    calib.estOpts = &estOpts;
    calib.jacInd = &jacInd;

    Mat9X calibJac(9, eStateSize);
    Mat96 prevPoseJac, nextPoseJac;
    Mat93 prevVelJac, nextVelJac;
    Mat92 gravityJac;

    Vec9 residual = evaluateInertialFactor(
        preint, calib, prevT, prevVel, nextT, nextVel, gravity,
        calibJac, prevPoseJac, prevVelJac, nextPoseJac, nextVelJac,
        gravityJac);

    if (!perturbNextState) {
      EXPECT_LT(residual.norm(), 1e-6)
          << "Trial " << trial << ": residual too large";
    }

    const double eps = 1e-7;

    // --- Numerical Jacobian: prev_pose (SE3, 6 columns) ---
    for (int i = 0; i < 6; i++) {
      Vec6 delta = Vec6::Zero();
      delta[i] = eps;
      SE3 tPlus = SE3::exp(delta) * prevT;
      delta[i] = -eps;
      SE3 tMinus = SE3::exp(delta) * prevT;

      Vec9 rPlus = evalResidualOnly(
          preint, calib, tPlus, prevVel, nextT, nextVel, gravity);
      Vec9 rMinus = evalResidualOnly(
          preint, calib, tMinus, prevVel, nextT, nextVel, gravity);

      Vec9 numCol = (rPlus - rMinus) / (2.0 * eps);
      Vec9 anlCol = prevPoseJac.col(i);
      double err =
          (numCol - anlCol).norm() / std::max(numCol.norm(), 1.0);
      maxErrPrevPose = std::max(maxErrPrevPose, err);
    }

    // --- Numerical Jacobian: prev_vel ---
    for (int i = 0; i < 3; i++) {
      Vec3 vPlus = prevVel, vMinus = prevVel;
      vPlus[i] += eps;
      vMinus[i] -= eps;

      Vec9 rPlus = evalResidualOnly(
          preint, calib, prevT, vPlus, nextT, nextVel, gravity);
      Vec9 rMinus = evalResidualOnly(
          preint, calib, prevT, vMinus, nextT, nextVel, gravity);

      Vec9 numCol = (rPlus - rMinus) / (2.0 * eps);
      Vec9 anlCol = prevVelJac.col(i);
      double err =
          (numCol - anlCol).norm() / std::max(numCol.norm(), 1.0);
      maxErrPrevVel = std::max(maxErrPrevVel, err);
    }

    // --- Numerical Jacobian: next_pose ---
    for (int i = 0; i < 6; i++) {
      Vec6 delta = Vec6::Zero();
      delta[i] = eps;
      SE3 tPlus = SE3::exp(delta) * nextT;
      delta[i] = -eps;
      SE3 tMinus = SE3::exp(delta) * nextT;

      Vec9 rPlus = evalResidualOnly(
          preint, calib, prevT, prevVel, tPlus, nextVel, gravity);
      Vec9 rMinus = evalResidualOnly(
          preint, calib, prevT, prevVel, tMinus, nextVel, gravity);

      Vec9 numCol = (rPlus - rMinus) / (2.0 * eps);
      Vec9 anlCol = nextPoseJac.col(i);
      double err =
          (numCol - anlCol).norm() / std::max(numCol.norm(), 1.0);
      maxErrNextPose = std::max(maxErrNextPose, err);
    }

    // --- Numerical Jacobian: next_vel ---
    for (int i = 0; i < 3; i++) {
      Vec3 vPlus = nextVel, vMinus = nextVel;
      vPlus[i] += eps;
      vMinus[i] -= eps;

      Vec9 rPlus = evalResidualOnly(
          preint, calib, prevT, prevVel, nextT, vPlus, gravity);
      Vec9 rMinus = evalResidualOnly(
          preint, calib, prevT, prevVel, nextT, vMinus, gravity);

      Vec9 numCol = (rPlus - rMinus) / (2.0 * eps);
      Vec9 anlCol = nextVelJac.col(i);
      double err =
          (numCol - anlCol).norm() / std::max(numCol.norm(), 1.0);
      maxErrNextVel = std::max(maxErrNextVel, err);
    }

    // --- Numerical Jacobian: gravity (S2, 2 columns) ---
    auto ortho =
        small_thing::S2::ortho(gravityVec); // 2x3
    for (int i = 0; i < 2; i++) {
      GravityData gPlus{
          .radius = gravity.radius,
          .vec = gravityVec + ortho.row(i).transpose() * eps};
      GravityData gMinus{
          .radius = gravity.radius,
          .vec = gravityVec - ortho.row(i).transpose() * eps};

      Vec9 rPlus = evalResidualOnly(
          preint, calib, prevT, prevVel, nextT, nextVel, gPlus);
      Vec9 rMinus = evalResidualOnly(
          preint, calib, prevT, prevVel, nextT, nextVel, gMinus);

      Vec9 numCol = (rPlus - rMinus) / (2.0 * eps);
      Vec9 anlCol = gravityJac.col(i);
      double err =
          (numCol - anlCol).norm() / std::max(numCol.norm(), 1.0);
      maxErrGravity = std::max(maxErrGravity, err);
    }

    // --- Numerical Jacobian: calibration ---
    const double calibEps = 1e-7;
    for (int i = 0; i < eStateSize; i++) {
      const bool isTimeOffset =
          i == jacInd.referenceImuTimeOffsetIdx() ||
          i == jacInd.gyroAccelTimeOffsetIdx();
      const double thisEps = isTimeOffset ? 1e-5 : calibEps;

      VecX delta = VecX::Zero(eStateSize);
      ImuCalibParam calibPlus = calib, calibMinus = calib;
      delta[i] = thisEps;
      calibPlus.boxPlus(delta);
      delta[i] = -thisEps;
      calibMinus.boxPlus(delta);

      Vec9 rPlus = evalResidualOnly(
          preint, calibPlus, prevT, prevVel, nextT, nextVel, gravity);
      Vec9 rMinus = evalResidualOnly(
          preint, calibMinus, prevT, prevVel, nextT, nextVel, gravity);

      Vec9 numCol = (rPlus - rMinus) / (2.0 * thisEps);
      Vec9 anlCol = calibJac.col(i);
      double err =
          (numCol - anlCol).norm() / std::max(numCol.norm(), 1.0);
      if (isTimeOffset) {
        maxErrCalibTime = std::max(maxErrCalibTime, err);
      } else {
        maxErrCalib = std::max(maxErrCalib, err);
      }
    }
  }

  std::cout << "\n=== Inertial Factor Jacobian Test [" << label << "] ===\n";
  auto tag = [](bool ok) { return ok ? "PASS" : "FAIL"; };
  bool prevPoseOk = maxErrPrevPose < 1e-5;
  bool prevVelOk = maxErrPrevVel < 1e-5;
  bool nextPoseOk = maxErrNextPose < 1e-5;
  bool nextVelOk = maxErrNextVel < 1e-5;
  bool gravityOk = maxErrGravity < 1e-5;
  bool calibOk = maxErrCalib < 1e-5;
  bool calibTimeOk = maxErrCalibTime < 1e-3;

  std::cout << "  [" << tag(prevPoseOk) << "] prev_pose:   " << maxErrPrevPose
            << "\n";
  std::cout << "  [" << tag(prevVelOk) << "] prev_vel:    " << maxErrPrevVel
            << "\n";
  std::cout << "  [" << tag(nextPoseOk) << "] next_pose:   " << maxErrNextPose
            << "\n";
  std::cout << "  [" << tag(nextVelOk) << "] next_vel:    " << maxErrNextVel
            << "\n";
  std::cout << "  [" << tag(gravityOk) << "] gravity:     " << maxErrGravity
            << "\n";
  std::cout << "  [" << tag(calibOk) << "] calib:       " << maxErrCalib
            << "\n";
  std::cout << "  [" << tag(calibTimeOk) << "] calib_time:  " << maxErrCalibTime
            << "\n";

  EXPECT_LT(maxErrPrevPose, 1e-5);
  EXPECT_LT(maxErrPrevVel, 1e-5);
  EXPECT_LT(maxErrNextPose, 1e-5);
  EXPECT_LT(maxErrNextVel, 1e-5);
  EXPECT_LT(maxErrGravity, 1e-5);
  EXPECT_LT(maxErrCalib, 1e-5);
  EXPECT_LT(maxErrCalibTime, 1e-3);
}

TEST(TestInertialFactor, JacobiansAtZero) {
  runJacobianTest(false);
}

TEST(TestInertialFactor, JacobiansAtNonzero) {
  runJacobianTest(true);
}

// ---------------------------------------------------------------------------
// TEST: Linearization stability
// ---------------------------------------------------------------------------
static void runLinearizationStabilityTest(bool perturbNextState) {
  const char* label =
      perturbNextState ? "nonzero-residual" : "zero-residual";
  std::mt19937 g(99);

  ImuCalibrationOptions estOpts(true);
  ImuCalibrationJacobianIndices jacInd(estOpts);
  ImuNoiseModelParameters noiseModel;
  const int eStateSize = estOpts.errorStateSize();

  const double targetCorrection = 1e-3;
  const double epsMax = 1e-3;
  const double epsMin = 1e-8;

  double maxDeltaRot = 0, maxDeltaVel = 0, maxDeltaPos = 0;
  double maxDeltaRotTime = 0, maxDeltaVelTime = 0, maxDeltaPosTime = 0;

  const int64_t timeStartUs = 850'000;
  const int64_t timeEndUs = 1'150'000;

  for (int trial = 0; trial < 30; trial++) {
    ImuMeasurementModelParameters imuParams = factoryImuParams();
    if (trial > 2) {
      boxPlus(imuParams, jacInd, randVecX(eStateSize, g) * 0.05);
    }
    if (trial % 2 == 1) {
      imuParams.dtReferenceAccelSec = imuParams.dtReferenceGyroSec;
    }

    auto meas = genImuMeasurements(g, 0, 2'000'000);
    auto preint = computePreIntegration(
        jacInd, meas, imuParams, noiseModel, timeStartUs, timeEndUs);

    // Random states
    const SO3 prevRot = SO3::exp(randVec<Vec3>(g));
    const Vec3 prevTrans = randVec<Vec3>(g) * 2.0;
    const SE3 prevT(prevRot, prevTrans);
    const Vec3 prevVel = randVec<Vec3>(g);

    Vec3 gDir = randVec<Vec3>(g);
    capNorm(gDir, 1.0);
    const Vec3 gravityVec = gDir.normalized() * 9.81;
    const GravityData gravity{.radius = gravityVec.norm(), .vec = gravityVec};

    auto [nextT, nextVel] =
        deriveNextState(preint, prevT, prevVel, gravityVec);

    if (perturbNextState) {
      Vec3 rotPert = randVec<Vec3>(g).normalized() * 0.087;
      Vec3 transPert = randVec<Vec3>(g).normalized() * 0.03;
      Vec6 se3Pert;
      se3Pert << transPert, rotPert;
      nextT = SE3::exp(se3Pert) * nextT;
      nextVel += randVec<Vec3>(g).normalized() * 0.03;
    }

    ImuCalibParam calib;
    calib.modelParams = imuParams;
    calib.estOpts = &estOpts;
    calib.jacInd = &jacInd;

    // Base residual
    Vec9 resBase = evalResidualOnly(
        preint, calib, prevT, prevVel, nextT, nextVel, gravity);

    // Perturb each calibration DOF
    for (int i = 0; i < eStateSize; i++) {
      const bool isTimeOffset =
          i == jacInd.referenceImuTimeOffsetIdx() ||
          i == jacInd.gyroAccelTimeOffsetIdx();

      // Adapt epsilon based on J column norm
      const double jColNorm = std::max(preint.J.col(i).norm(), 1e-10);
      const double epsCap = isTimeOffset ? 1e-7 : epsMax;
      const double thisEps =
          std::clamp(targetCorrection / jColNorm, epsMin, epsCap);

      // Skip DOFs where expected second-order error is too large
      const double expectedErrPerEps = jColNorm * jColNorm * thisEps;
      if (expectedErrPerEps > 0.1) {
        continue;
      }

      VecX delta = VecX::Zero(eStateSize);
      delta[i] = thisEps;
      ImuMeasurementModelParameters perturbedParams = imuParams;
      boxPlus(perturbedParams, jacInd, delta);

      // Build preintegration at perturbed linearization point
      auto perturbedPreint = computePreIntegration(
          jacInd, meas, perturbedParams, noiseModel, timeStartUs, timeEndUs);

      // Evaluate at ORIGINAL imu_params — correction should compensate
      Vec9 resPert = evalResidualOnly(
          perturbedPreint, calib, prevT, prevVel, nextT, nextVel, gravity);

      Vec9 deltaRes = resPert - resBase;
      double dRot = deltaRes.head<3>().norm() / thisEps;
      double dVel = deltaRes.segment<3>(3).norm() / thisEps;
      double dPos = deltaRes.tail<3>().norm() / thisEps;

      if (isTimeOffset) {
        maxDeltaRotTime = std::max(maxDeltaRotTime, dRot);
        maxDeltaVelTime = std::max(maxDeltaVelTime, dVel);
        maxDeltaPosTime = std::max(maxDeltaPosTime, dPos);
      } else {
        maxDeltaRot = std::max(maxDeltaRot, dRot);
        maxDeltaVel = std::max(maxDeltaVel, dVel);
        maxDeltaPos = std::max(maxDeltaPos, dPos);
      }
    }
  }

  std::cout << "\n=== Linearization Stability [" << label << "] ===\n";
  auto tag = [](bool ok) { return ok ? "PASS" : "FAIL"; };
  bool rotOk = maxDeltaRot < 2e-1;
  bool velOk = maxDeltaVel < 2e-1;
  bool posOk = maxDeltaPos < 2e-1;
  bool rotTimeOk = maxDeltaRotTime < 2e-1;
  bool velTimeOk = maxDeltaVelTime < 2e-1;
  bool posTimeOk = maxDeltaPosTime < 2e-1;

  std::cout << "  [" << tag(rotOk) << "] rot:       " << maxDeltaRot << "\n";
  std::cout << "  [" << tag(velOk) << "] vel:       " << maxDeltaVel << "\n";
  std::cout << "  [" << tag(posOk) << "] pos:       " << maxDeltaPos << "\n";
  std::cout << "  [" << tag(rotTimeOk) << "] rot_time:  " << maxDeltaRotTime
            << "\n";
  std::cout << "  [" << tag(velTimeOk) << "] vel_time:  " << maxDeltaVelTime
            << "\n";
  std::cout << "  [" << tag(posTimeOk) << "] pos_time:  " << maxDeltaPosTime
            << "\n";

  EXPECT_LT(maxDeltaRot, 2e-1);
  EXPECT_LT(maxDeltaVel, 2e-1);
  EXPECT_LT(maxDeltaPos, 2e-1);
  EXPECT_LT(maxDeltaRotTime, 2e-1);
  EXPECT_LT(maxDeltaVelTime, 2e-1);
  EXPECT_LT(maxDeltaPosTime, 2e-1);
}

TEST(TestInertialFactor, LinearizationStabilityAtZero) {
  runLinearizationStabilityTest(false);
}

TEST(TestInertialFactor, LinearizationStabilityAtNonzero) {
  runLinearizationStabilityTest(true);
}
