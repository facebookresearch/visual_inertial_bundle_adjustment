/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/common/Enumerate.h>
#include <viba/problem/SingleSessionProblem.h>

#include <logging/Checks.h>
#define DEFAULT_LOG_CHANNEL "ViBa::InertialFactor"
#include <logging/Log.h>

namespace visual_inertial_ba {

// Note: don't put in anonymous namespace so we can find the factors via introspection
// @lint-ignore-every CLANGTIDY facebook-hte-ShadowingClass
struct InertialFactor {
  explicit InertialFactor(const PreIntegrationData& preintegrationData)
      : preintegrationData_(preintegrationData) {}

  Vec9 operator()(
      const ImuCalibParam& calib,
      const SE3& prev_T_imu_world,
      const Vec3& prev_vel_world,
      const SE3& next_T_imu_world,
      const Vec3& next_vel_world,
      const GravityData& gravityWorld,
      Ref<Mat9X> calib_Jacobian,
      Ref<Mat96> prev_T_imu_world_Jacobian,
      Ref<Mat93> prev_vel_world_Jacobian,
      Ref<Mat96> next_T_imu_world_Jacobian,
      Ref<Mat93> next_vel_world_Jacobian,
      Ref<Mat92> gravityWorld_Jacobian) const {
    // imu calib when preintegration was computed
    const int preintErrorSize = calib.jacInd->getErrorStateSize();
    const double deltaTimeSec = preintegrationData_.rvp.dtSec;

    // compute correction as consequence of variation in calibration
    // estimate from printegration: (calib - preintegationEvaluationCalib)
    Map<VecX> deltaCalib((double*)alloca(sizeof(double) * preintErrorSize), preintErrorSize);
    calib.boxMinus(preintegrationData_.calibEvalPoint, deltaCalib);
    const Vec9 preintCorrection = preintegrationData_.J * deltaCalib;

    // rotation error
    const SO3 R_rotCorrection = SO3::exp(-preintCorrection.head<3>());
    const SO3 corrected_R_ImuNext_ImuPrev = R_rotCorrection * preintegrationData_.rvp.R.inverse();
    const SO3 R_rotErr =
        corrected_R_ImuNext_ImuPrev * prev_T_imu_world.so3() * next_T_imu_world.so3().inverse();
    const Vec3 logRotErr = -R_rotErr.log();

    // velocity error and related jacobians
    const Vec3 deltaVelocity_world_est =
        next_vel_world - prev_vel_world - gravityWorld.vec * deltaTimeSec;
    const Vec3 deltaVelocity_prevImu_est = prev_T_imu_world.so3() * deltaVelocity_world_est;
    const Vec3 velErr =
        preintegrationData_.rvp.dV - deltaVelocity_prevImu_est + preintCorrection.segment<3>(3);

    // position error and related jacobians
    const SO3 R_prevImu_nextImu = prev_T_imu_world.so3() * next_T_imu_world.so3().inverse();
    const Vec3 deltaPosition_prevImu_est = prev_T_imu_world.translation() //
        - R_prevImu_nextImu * next_T_imu_world.translation() //
        - prev_T_imu_world.so3() *
            (prev_vel_world * deltaTimeSec +
             gravityWorld.vec * (0.5 * deltaTimeSec * deltaTimeSec));
    const Vec3 posErr =
        preintegrationData_.rvp.dP - deltaPosition_prevImu_est + preintCorrection.segment<3>(6);

    const Mat33 dLogRotErr_dLeftRotError = SO3::leftJacobianInverse(-logRotErr);

    // clang-format off
    if (!isNull(prev_T_imu_world_Jacobian)) {
      prev_T_imu_world_Jacobian <<                                                         //
        Mat33::Zero(),      -dLogRotErr_dLeftRotError * corrected_R_ImuNext_ImuPrev.Adj(), //
        Mat33::Zero(),      -SO3::hat(-deltaVelocity_prevImu_est),                         //
        -Mat33::Identity(), -SO3::hat(-deltaPosition_prevImu_est);
    }
    if (!isNull(prev_vel_world_Jacobian)) {
      const Mat33 dPrevImuPt_dWorldPt = prev_T_imu_world.so3().matrix();
      prev_vel_world_Jacobian <<            //
        Mat33::Zero(),                      //
        dPrevImuPt_dWorldPt,                //
        dPrevImuPt_dWorldPt * deltaTimeSec;
    }
    if (!isNull(next_T_imu_world_Jacobian)) {
      next_T_imu_world_Jacobian <<                                //
        Mat33::Zero(), dLogRotErr_dLeftRotError * R_rotErr.Adj(), //
        Mat36::Zero(),                                            //
        R_prevImu_nextImu.matrix(), Mat33::Zero();
    }
    if (!isNull(next_vel_world_Jacobian)) {
      next_vel_world_Jacobian <<          //
        Mat33::Zero(),                    //
        -prev_T_imu_world.so3().matrix(), //
        Mat33::Zero();
    }
    if (!isNull(gravityWorld_Jacobian)) {
      Mat32 dVel_dG = deltaTimeSec * prev_T_imu_world.so3().matrix() *
          small_thing::S2::ortho(gravityWorld.vec).transpose();
      gravityWorld_Jacobian <<          //
        Mat32::Zero(),                  //
        dVel_dG,                        //
        dVel_dG * (0.5 * deltaTimeSec);
    }
    // clang-format on
    if (!isNull(calib_Jacobian)) {
      const Mat33 dLogRotErr_dResidualCorrection =
          dLogRotErr_dLeftRotError * SO3::leftJacobian(-preintCorrection.head<3>());
      const int calibErrorSize = calib.estOpts->errorStateSize();
      XR_CHECK_EQ(calib_Jacobian.cols(), calibErrorSize);

      // if optimizing a subset of the preint's parameters, extract the sub-Jacobian
      const auto& calibJac = preintegrationData_.J;
      XR_CHECK_EQ(calibErrorSize, calibJac.cols());

      calib_Jacobian << dLogRotErr_dResidualCorrection * calibJac.topRows<3>(),
          calibJac.bottomRows<6>();
    }
    Vec9 rotVelPosError;
    rotVelPosError << logRotErr, velErr, posErr;
    return rotVelPosError;
  }

 private:
  const PreIntegrationData& preintegrationData_;
};

// Note: don't put in anonymous namespace so we can find the factors via introspection
// @lint-ignore-every CLANGTIDY facebook-hte-ShadowingClass
struct SecondaryImuInertialFactor {
  explicit SecondaryImuInertialFactor(const PreIntegrationData& preintegrationData)
      : inertialFactor_(preintegrationData) {}

  /* secondary state allows to compute imu's pose/vel from body imu's params and extrinsics */
  struct SecondaryState {
    SecondaryState(
        const SE3& T_bodyImu_world,
        const Vec3& bodyImu_vel_world,
        const Vec3& bodyImu_omega,
        const SE3& T_imu_bodyImu) {
      t_bodyImu_imu = T_imu_bodyImu.inverse().translation();
      imu_vel_bodyImu = bodyImu_omega.cross(t_bodyImu_imu);
      R_world_bodyImu = T_bodyImu_world.so3().inverse();
      T_imu_world = T_imu_bodyImu * T_bodyImu_world;
      imu_vel_world = bodyImu_vel_world + R_world_bodyImu * imu_vel_bodyImu;
    }

    /* trace back jacobian from imu to body imu */
    void composeJacobians(
        const Vec3& bodyImu_omega,
        const SE3& T_imu_bodyImu,
        const Mat96& T_imu_world_Jacobian,
        const Mat93& imu_vel_world_Jacobian,
        Ref<Mat96> T_bodyImu_world_Jacobian,
        Ref<Mat93> bodyImu_vel_world_Jacobian,
        Ref<Mat93> bodyImu_omega_Jacobian,
        Ref<Mat96> T_imu_bodyImu_Jacobian) {
      if (!isNull(T_bodyImu_world_Jacobian)) {
        Mat36 dVelWorld_d_T_bodyImu_world;
        dVelWorld_d_T_bodyImu_world << Mat33::Zero(),
            R_world_bodyImu.Adj() * (-SO3::hat(-imu_vel_bodyImu));
        T_bodyImu_world_Jacobian = T_imu_world_Jacobian * T_imu_bodyImu.Adj() +
            imu_vel_world_Jacobian * dVelWorld_d_T_bodyImu_world;
      }
      if (!isNull(bodyImu_vel_world_Jacobian)) {
        bodyImu_vel_world_Jacobian = imu_vel_world_Jacobian;
      }
      if (!isNull(bodyImu_omega_Jacobian)) {
        bodyImu_omega_Jacobian =
            imu_vel_world_Jacobian * (R_world_bodyImu.Adj() * SO3::hat(-t_bodyImu_imu));
      }
      if (!isNull(T_imu_bodyImu_Jacobian)) {
        Mat36 dVelWorld_d_T_imu_bodyImu;
        dVelWorld_d_T_imu_bodyImu << R_world_bodyImu.Adj() * SO3::hat(bodyImu_omega) *
                (-T_imu_bodyImu.so3().Adj().transpose()),
            Mat33::Zero();
        T_imu_bodyImu_Jacobian =
            T_imu_world_Jacobian + imu_vel_world_Jacobian * dVelWorld_d_T_imu_bodyImu;
      }
    }

    Vec3 t_bodyImu_imu;
    Vec3 imu_vel_bodyImu;
    SO3 R_world_bodyImu;
    SE3 T_imu_world;
    Vec3 imu_vel_world;
  };

  Vec9 operator()(
      const ImuCalibParam& calib,
      const SE3& prev_T_bodyImu_world,
      const Vec3& prev_bodyImu_vel_world,
      const Vec3& prev_bodyImu_omega,
      const SE3& prev_T_imu_bodyImu,
      const SE3& next_T_bodyImu_world,
      const Vec3& next_bodyImu_vel_world,
      const Vec3& next_bodyImu_omega,
      const SE3& next_T_imu_bodyImu,
      const GravityData& gravityWorld,
      Ref<Mat9X> calib_Jacobian,
      Ref<Mat96> prev_T_bodyImu_world_Jacobian,
      Ref<Mat93> prev_bodyImu_vel_world_Jacobian,
      Ref<Mat93> prev_bodyImu_omega_Jacobian,
      Ref<Mat96> prev_T_imu_bodyImu_Jacobian,
      Ref<Mat96> next_T_bodyImu_world_Jacobian,
      Ref<Mat93> next_bodyImu_vel_world_Jacobian,
      Ref<Mat93> next_bodyImu_omega_Jacobian,
      Ref<Mat96> next_T_imu_bodyImu_Jacobian,
      Ref<Mat92> gravityWorld_Jacobian) const {
    SecondaryState prevState(
        prev_T_bodyImu_world, prev_bodyImu_vel_world, prev_bodyImu_omega, prev_T_imu_bodyImu);
    SecondaryState nextState(
        next_T_bodyImu_world, next_bodyImu_vel_world, next_bodyImu_omega, next_T_imu_bodyImu);

    Mat96 prev_T_imu_world_Jacobian;
    Mat93 prev_imu_vel_world_Jacobian;
    Mat96 next_T_imu_world_Jacobian;
    Mat93 next_imu_vel_world_Jacobian;
    Vec9 rotVelPosError = inertialFactor_(
        calib,
        prevState.T_imu_world,
        prevState.imu_vel_world,
        nextState.T_imu_world,
        nextState.imu_vel_world,
        gravityWorld,
        calib_Jacobian,
        prev_T_imu_world_Jacobian,
        prev_imu_vel_world_Jacobian,
        next_T_imu_world_Jacobian,
        next_imu_vel_world_Jacobian,
        gravityWorld_Jacobian);

    prevState.composeJacobians(
        prev_bodyImu_omega,
        prev_T_imu_bodyImu,
        prev_T_imu_world_Jacobian,
        prev_imu_vel_world_Jacobian,
        prev_T_bodyImu_world_Jacobian,
        prev_bodyImu_vel_world_Jacobian,
        prev_bodyImu_omega_Jacobian,
        prev_T_imu_bodyImu_Jacobian);
    nextState.composeJacobians(
        next_bodyImu_omega,
        next_T_imu_bodyImu,
        next_T_imu_world_Jacobian,
        next_imu_vel_world_Jacobian,
        next_T_bodyImu_world_Jacobian,
        next_bodyImu_vel_world_Jacobian,
        next_bodyImu_omega_Jacobian,
        next_T_imu_bodyImu_Jacobian);

    return rotVelPosError;
  }

  Vec9 operator()(
      const ImuCalibParam& calib,
      const SE3& prev_T_bodyImu_world,
      const Vec3& prev_bodyImu_vel_world,
      const Vec3& prev_bodyImu_omega,
      const SE3& next_T_bodyImu_world,
      const Vec3& next_bodyImu_vel_world,
      const Vec3& next_bodyImu_omega,
      const SE3& common_T_imu_bodyImu,
      const GravityData& gravityWorld,
      Ref<Mat9X> calib_Jacobian,
      Ref<Mat96> prev_T_bodyImu_world_Jacobian,
      Ref<Mat93> prev_bodyImu_vel_world_Jacobian,
      Ref<Mat93> prev_bodyImu_omega_Jacobian,
      Ref<Mat96> next_T_bodyImu_world_Jacobian,
      Ref<Mat93> next_bodyImu_vel_world_Jacobian,
      Ref<Mat93> next_bodyImu_omega_Jacobian,
      Ref<Mat96> common_T_imu_bodyImu_Jacobian,
      Ref<Mat92> gravityWorld_Jacobian) const {
    Mat96 prev_T_imu_bodyImu_Jacobian, next_T_imu_bodyImu_Jacobian;
    Vec9 rotVelPosError = this->operator()(
        calib,
        prev_T_bodyImu_world,
        prev_bodyImu_vel_world,
        prev_bodyImu_omega,
        common_T_imu_bodyImu,
        next_T_bodyImu_world,
        next_bodyImu_vel_world,
        next_bodyImu_omega,
        common_T_imu_bodyImu,
        gravityWorld,
        calib_Jacobian,
        prev_T_bodyImu_world_Jacobian,
        prev_bodyImu_vel_world_Jacobian,
        prev_bodyImu_omega_Jacobian,
        isNull(common_T_imu_bodyImu_Jacobian) ? NullRef() : Ref<Mat96>(prev_T_imu_bodyImu_Jacobian),
        next_T_bodyImu_world_Jacobian,
        next_bodyImu_vel_world_Jacobian,
        next_bodyImu_omega_Jacobian,
        isNull(common_T_imu_bodyImu_Jacobian) ? NullRef() : Ref<Mat96>(next_T_imu_bodyImu_Jacobian),
        gravityWorld_Jacobian);
    if (!isNull(common_T_imu_bodyImu_Jacobian)) {
      common_T_imu_bodyImu_Jacobian = prev_T_imu_bodyImu_Jacobian + next_T_imu_bodyImu_Jacobian;
    }
    return rotVelPosError;
  }

 private:
  InertialFactor inertialFactor_;
};

void SingleSessionProblem::addInertialFactor(
    int64_t prevRigIndex,
    int64_t nextRigIndex,
    int imuIndex,
    const PreIntegrationData& preintegrationData,
    const SoftLossType& imuErrorLoss) {
  const Mat99 H = preintegrationData.rvpCov.llt().solve(Mat99::Identity()); // information matrix
  auto& prevPoseVar = inertialPose(prevRigIndex);
  auto& nextPoseVar = inertialPose(nextRigIndex);
  auto& calibVar = imuCalib(prevRigIndex, imuIndex).var;
  small_thing::FactorStoreBase* factorStorePtr;
  int64_t factorIndex;

  XR_CHECK_EQ(preintegrationData.J.cols(), calibVar.getTangentDim());

  if (imuIndex == 0) {
    std::tie(factorStorePtr, factorIndex) = opt_.addFactor(
        InertialFactor(preintegrationData),
        H,
        imuErrorLoss,
        calibVar,
        prevPoseVar.T_bodyImu_world,
        prevPoseVar.vel_world,
        nextPoseVar.T_bodyImu_world,
        nextPoseVar.vel_world,
        *gravityWorld_);
  } else { // imuIndex >= 1
    const int64_t prevImuExtrinsicsIndex = findOrDie(rigImuToExtrIndex_, {prevRigIndex, imuIndex});
    const int64_t nextImuExtrinsicsIndex = findOrDie(rigImuToExtrIndex_, {nextRigIndex, imuIndex});
    if (prevImuExtrinsicsIndex == nextImuExtrinsicsIndex) {
      auto& common_T_Imu_BodyImu_var = Ts_Imu_BodyImu_[prevImuExtrinsicsIndex].var;

      std::tie(factorStorePtr, factorIndex) = opt_.addFactor(
          SecondaryImuInertialFactor(preintegrationData),
          H,
          imuErrorLoss,
          calibVar,
          prevPoseVar.T_bodyImu_world,
          prevPoseVar.vel_world,
          prevPoseVar.omega,
          nextPoseVar.T_bodyImu_world,
          nextPoseVar.vel_world,
          nextPoseVar.omega,
          common_T_Imu_BodyImu_var,
          *gravityWorld_);
    } else {
      auto& prev_T_Imu_BodyImu_var = Ts_Imu_BodyImu_[prevImuExtrinsicsIndex].var;
      auto& next_T_Imu_BodyImu_var = Ts_Imu_BodyImu_[nextImuExtrinsicsIndex].var;

      std::tie(factorStorePtr, factorIndex) = opt_.addFactor(
          SecondaryImuInertialFactor(preintegrationData),
          H,
          imuErrorLoss,
          calibVar,
          prevPoseVar.T_bodyImu_world,
          prevPoseVar.vel_world,
          prevPoseVar.omega,
          prev_T_Imu_BodyImu_var,
          nextPoseVar.T_bodyImu_world,
          nextPoseVar.vel_world,
          nextPoseVar.omega,
          next_T_Imu_BodyImu_var,
          *gravityWorld_);
    }
  }

  if (verbosity_ == Verbose) {
    Vec9 rotVelPosErr;
    factorStorePtr->unweightedError(factorIndex, rotVelPosErr);
    XR_LOGI("IMU[{},{}..{}], RES: {}", imuIndex, prevRigIndex, nextRigIndex, rotVelPosErr);
  }
  return;
}

} // namespace visual_inertial_ba
