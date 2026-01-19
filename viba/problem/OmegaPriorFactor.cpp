/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/common/Enumerate.h>
#include <viba/problem/SingleSessionProblem.h>

#define DEFAULT_LOG_CHANNEL "ViBa::RandomWalkFactors"
#include <logging/Log.h>

namespace visual_inertial_ba {

void SingleSessionProblem::addOmegaPriorFactor(
    int64_t rigIndex,
    int imuIndex,
    const Vec3& omegaRadSec_Imu) {
  small_thing::FactorStoreBase* factorStorePtr;
  int64_t factorIndex;
  auto& rigVar = findOrDie(inertialPoses_, rigIndex);
  if (imuIndex == 0) {
    std::tie(factorStorePtr, factorIndex) = opt_.addFactor(
        [omegaRadSec_Imu](const Vec3& omega, Ref<Mat33>&& omegaJacobian) -> Vec3 {
          if (!isNull(omegaJacobian)) {
            omegaJacobian = Mat33::Identity() / kMultiImuOmegaPriorStdRadSec;
          }
          return (omega - omegaRadSec_Imu) / kMultiImuOmegaPriorStdRadSec;
        },
        rigVar.omega);
  } else {
    auto& T_Imu_BodyImu_var = T_Imu_BodyImu(rigIndex, imuIndex);

    std::tie(factorStorePtr, factorIndex) = opt_.addFactor(
        [omegaRadSec_Imu](
            const Vec3& omega,
            const SE3& T_Imu_BodyImu,
            Ref<Mat33>&& omegaJacobian,
            Ref<Mat36>&& T_Imu_BodyImu_Jacobian) -> Vec3 {
          if (!isNull(omegaJacobian)) {
            omegaJacobian = Mat33::Identity() / kMultiImuOmegaPriorStdRadSec;
          }
          Vec3 omegaAtEndRadSec_BodyImu = T_Imu_BodyImu.so3().inverse() * omegaRadSec_Imu;
          if (!isNull(T_Imu_BodyImu_Jacobian)) {
            T_Imu_BodyImu_Jacobian << Mat33::Zero(),
                -T_Imu_BodyImu.so3().Adj().transpose() * (-SO3::hat(-omegaRadSec_Imu)) /
                kMultiImuOmegaPriorStdRadSec;
          }
          return (omega - omegaAtEndRadSec_BodyImu) / kMultiImuOmegaPriorStdRadSec;
        },
        rigVar.omega,
        T_Imu_BodyImu_var.var);
  }

  if (verbosity_ == Verbose) {
    Vec3 omegaErr;
    factorStorePtr->unweightedError(factorIndex, omegaErr);
    XR_LOGI(
        "Omega[{},{}]: RES: {} (degrees)", rigIndex, imuIndex, radiansToDegrees(omegaErr.norm()));
  }
}

} // namespace visual_inertial_ba
