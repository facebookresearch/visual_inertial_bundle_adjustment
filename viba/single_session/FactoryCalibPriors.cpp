/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <camera_model/RandomWalkCov.h>
#include <extrinsics_model/RandomWalkCov.h>
#include <imu_model/RandomWalkCov.h>
#include <viba/single_session/SingleSessionAdapter.h>

#define DEFAULT_LOG_CHANNEL "ViBa::FactoryCalibPriors"
#include <logging/Log.h>

namespace visual_inertial_ba {

using RigSensorToIndex = SingleSessionProblem::RigSensorToIndex;

// Util: count how many times a calib variable is referenced by KRs
// Returns map {calibVarIndex -> (count, sensorIndex)}
static auto getParamToCountSensorIndex(const RigSensorToIndex& rigSensorToCalibIndex) {
  std::unordered_map<int64_t, std::pair<int, int>> refCounts;
  for (auto [rigIndexSensorIndex, varIndex] : rigSensorToCalibIndex) {
    auto [rigIndex, sensorIndex] = rigIndexSensorIndex;
    auto& ref = refCounts[varIndex];
    ref.first++;
    ref.second = sensorIndex; // save sensor index
  }
  return refCounts;
}

void SingleSessionAdapter::addCamIntrFactoryCalibPriors(double stdDevInflate) {
  if (verbosity_ != Muted) {
    XR_LOGI("Adding factory-priors on imu calibs (nimus: {})", numImus_);
  }
  auto refCounts = getParamToCountSensorIndex(prob_.rigCamToModelIndex());

  int64_t nAdded = 0;
  for (auto& [calibIndex, countAndIndex] : refCounts) {
    auto [count, camIndex] = countAndIndex;

    const int fIdx = matcher_.slamCamIndexToFactoryCalib[camIndex];
    CameraModelParam factoryModelParams =
        fData_.factoryCalibration.calib.getConvertedCameraModelParam(fIdx);
    const CameraModelParam& calib = prob_.cameraModel_at(calibIndex).var.value;

    // SANITY check: the first param (fx or unique focal length) should not be too far.
    // if that is the case most likely the camera calibration is incorrectly adapted to resolution
    constexpr double kFocalLEngthMaxRelError = 0.1; // a generous 10%
    const double onlineCalibF = calib.intrinsicParams()[0];
    const double factoryCalibF = factoryModelParams.intrinsicParams()[0];
    XR_CHECK_LT(
        std::abs(factoryCalibF - onlineCalibF) / factoryCalibF,
        kFocalLEngthMaxRelError,
        "\nCamera n. {}: factory calibration prior params\n  {}\n"
        "are very different from online calibration params\n  {}\n"
        "This is likely due to incorrectly adapted camera calibration,\n"
        "make sure that online and factory calibration params refer to\n"
        "the same image resolution.",
        camIndex,
        factoryModelParams.intrinsicParams(),
        calib.intrinsicParams());

    VecX turnOnStdDev = cameraModelTurnOnStdDev(calib);
    turnOnStdDev *= stdDevInflate;
    VecX H = turnOnStdDev.cwiseProduct(turnOnStdDev).cwiseInverse();
    H *= count;

    prob_.addCamIntrinsicsPrior(calibIndex, factoryModelParams, H);
  }
  if (verbosity_ != Muted) {
    XR_LOGI("Added factory-priors on imu calibs: {}", nAdded);
  }
}

// Camera extrinsics turn-on standard deviations
constexpr double kCamExtrinsicPosTurnonStdM = 4e-4; // translation: 0.4mm
constexpr double kCamExtrinsicRotTurnonStdRad = 0.2 * (M_PI / 180); // rotation: 0.2degs

void SingleSessionAdapter::addCamExtrFactoryCalibPriors(double stdDevInflate) {
  auto refCounts = getParamToCountSensorIndex(prob_.rigCamToExtrIndex());

  for (auto& [calibIndex, ref] : refCounts) {
    auto [count, camIndex] = ref;

    const int fIdx = matcher_.slamCamIndexToFactoryCalib[camIndex];
    const SE3& factoryT_Cam_BodyImu = fData_.factoryCalibration.calib.T_Cam_BodyImu[fIdx];

    VecX turnOnStdDev(6);
    turnOnStdDev.head(3).setConstant(kCamExtrinsicPosTurnonStdM);
    turnOnStdDev.tail(3).setConstant(degreesToRadians(kCamExtrinsicRotTurnonStdRad));

    turnOnStdDev *= stdDevInflate;
    VecX H = turnOnStdDev.cwiseProduct(turnOnStdDev).cwiseInverse();
    H *= count;

    prob_.addCamExtrinsicsPrior(calibIndex, factoryT_Cam_BodyImu, H);
  }
}

// Imu extrinsics: we use values from imu's factory calibration
void SingleSessionAdapter::addImuExtrFactoryCalibPriors(double stdDevInflate) {
  auto refCounts = getParamToCountSensorIndex(prob_.rigImuToExtrIndex());

  for (auto& [calibIndex, ref] : refCounts) {
    auto [count, imuIndex] = ref;

    const int fIdx = matcher_.slamImuIndexToFactoryCalib[imuIndex];
    const SE3& factoryT_Imu_BodyImu = fData_.factoryCalibration.calib.T_Imu_BodyImu[fIdx];
    const auto& noiseParams = fData_.factoryCalibration.imuNoiseModels[fIdx];

    VecX turnOnStdDev(6);
    turnOnStdDev.head(3) = noiseParams.imuBodyImuTurnonPosStdM;
    turnOnStdDev.tail(3) = noiseParams.imuBodyImuTurnonRotStdRad;

    turnOnStdDev *= stdDevInflate;
    VecX H = turnOnStdDev.cwiseProduct(turnOnStdDev).cwiseInverse();
    H *= count;

    prob_.addImuExtrinsicsPrior(calibIndex, factoryT_Imu_BodyImu, H);
  }
}

// Imu intrinsics: we use values from imu's factory calibration
void SingleSessionAdapter::addImuFactoryCalibPriors(double stdDevInflate) {
  auto refCounts = getParamToCountSensorIndex(prob_.rigImuToCalibIndex());

  for (auto& [calibIndex, ref] : refCounts) {
    auto [count, imuIndex] = ref;

    const ImuCalibParam& calib = prob_.imuCalib_at(calibIndex).var.value;
    const int fIdx = matcher_.slamImuIndexToFactoryCalib[imuIndex];
    const auto& factoryModelParams = fData_.factoryCalibration.calib.imuModelParameters[fIdx];
    const auto& noiseParams = fData_.factoryCalibration.imuNoiseModels[fIdx];

    VecX turnOnStdDev = imuCalibTurnonStdDev(noiseParams, *calib.jacInd);

    turnOnStdDev *= stdDevInflate;
    VecX H = turnOnStdDev.cwiseProduct(turnOnStdDev).cwiseInverse();
    H *= count;

    prob_.addImuPrior(calibIndex, factoryModelParams, H);
  }
}

} // namespace visual_inertial_ba
