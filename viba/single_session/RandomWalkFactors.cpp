/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
#include <camera_model/RandomWalkCov.h>
#include <extrinsics_model/RandomWalkCov.h>
#include <imu_model/RandomWalkCov.h>
#include <viba/common/Enumerate.h>
#include <viba/single_session/SingleSessionAdapter.h>

#define DEFAULT_LOG_CHANNEL "ViBa::RandomWalkFactors"
#include <logging/Log.h>

namespace visual_inertial_ba {

void SingleSessionAdapter::addAllRandomWalkFactors(
    double imuRWinflate,
    double camIntrRWinflate,
    double imuExtrRWinflate,
    double camExtrRWinflate) {
  addImuRWFactors(imuRWinflate);

  addCamIntrinsicsRWFactors(camIntrRWinflate);

  if (numImus_ > 1) {
    addImuExtrinsicsRWFactors(imuExtrRWinflate);
  }

  addCamExtrinsicsRWFactors(camExtrRWinflate);
}

void SingleSessionAdapter::addImuRWFactors(double imuRWinflate) {
  if (verbosity_ != Muted) {
    XR_LOGI("Adding RW on imu calibs (nimus: {})", numImus_);
  }

  int64_t nAdded = 0;
  const int numVariables = prob_.numImuCalibVariables();
  for (size_t i = 0; i < numVariables; i++) {
    const auto& thisVar = prob_.imuCalib_at(i);
    XR_CHECK_GE(thisVar.sensorIndex, 0);
    if (thisVar.prevVarIndex < 0) {
      continue;
    }
    const auto& prevVar = prob_.imuCalib_at(thisVar.prevVarIndex);

    const int fIdx = matcher_.slamImuIndexToFactoryCalib[thisVar.sensorIndex];
    const auto& noiseParams = fData_.factoryCalibration.imuNoiseModels[fIdx];
    VecX diagCov = imuCalibRandomWalkCov(
        noiseParams,
        *thisVar.var.value.jacInd,
        (thisVar.averageTimestampUs - prevVar.averageTimestampUs) * 1e-6);
    diagCov *= imuRWinflate;
    VecX diagSqrtH = diagCov.cwiseInverse().cwiseSqrt();
    prob_.addImuCalibRWFactor(thisVar.prevVarIndex, i, diagSqrtH);
    nAdded++;
  }
  if (verbosity_ != Muted) {
    XR_LOGI("Added RW on imu calibs: {}", nAdded);
  }
}

void SingleSessionAdapter::addCamIntrinsicsRWFactors(double camIntrRWinflate) {
  if (verbosity_ != Muted) {
    XR_LOGI("Adding RW on camera intrinsics (ncams: {})", numCameras_);
  }

  int64_t nAdded = 0;
  const int numVariables = prob_.numCameraModelVariables();
  for (size_t i = 0; i < numVariables; i++) {
    const auto& thisVar = prob_.cameraModel_at(i);
    XR_CHECK_GE(thisVar.sensorIndex, 0);
    if (thisVar.prevVarIndex < 0) {
      continue;
    }
    const auto& prevVar = prob_.cameraModel_at(thisVar.prevVarIndex);

    VecX diagCov = cameraModelRandomWalkCov(
        thisVar.var.value,
        /* dtSec = */ (thisVar.averageTimestampUs - prevVar.averageTimestampUs) * 1e-6
        /* deltaTempC = */);
    diagCov *= camIntrRWinflate;
    VecX diagSqrtH = diagCov.cwiseInverse().cwiseSqrt();
    prob_.addCamIntrRWFactor(thisVar.prevVarIndex, i, diagSqrtH);
    nAdded++;
  }
  if (verbosity_ != Muted) {
    XR_LOGI("Added RW on camera intrinsics: {}", nAdded);
  }
}

void SingleSessionAdapter::addImuExtrinsicsRWFactors(double imuExtrRWinflate) {
  if (verbosity_ != Muted) {
    XR_LOGI("Adding RW on imu extrinsics (nimus: {})", numImus_);
  }

  int64_t nAdded = 0;
  const int numVariables = prob_.numImuExtrinsicsVariables();
  for (size_t i = 0; i < numVariables; i++) {
    const auto& thisVar = prob_.T_Imu_BodyImu_at(i);
    XR_CHECK_GE(thisVar.sensorIndex, 0);
    if (thisVar.prevVarIndex < 0) {
      continue;
    }
    const auto& prevVar = prob_.T_Imu_BodyImu_at(thisVar.prevVarIndex);

    const int fIdx = matcher_.slamImuIndexToFactoryCalib[thisVar.sensorIndex];
    const auto& noiseParams = fData_.factoryCalibration.imuNoiseModels[fIdx];
    VecX diagCov = imuExtrRandomWalkCov(
        noiseParams,
        /* dtSec = */ (thisVar.averageTimestampUs - prevVar.averageTimestampUs) * 1e-6);
    diagCov *= imuExtrRWinflate;
    VecX diagSqrtH = diagCov.cwiseInverse().cwiseSqrt();
    prob_.addImuExtrRWFactor(thisVar.prevVarIndex, i, diagSqrtH);
    nAdded++;
  }
  if (verbosity_ != Muted) {
    XR_LOGI("Added RW on imu extrinsics: {}", nAdded);
  }
}

void SingleSessionAdapter::addCamExtrinsicsRWFactors(double camExtrRWinflate) {
  if (verbosity_ != Muted) {
    XR_LOGI("Adding RW on camera extrinsics (ncams: {})", numCameras_);
  }

  int64_t nAdded = 0;
  const int numVariables = prob_.numCameraExtrinsicsVariables();
  for (size_t i = 0; i < numVariables; i++) {
    const auto& thisVar = prob_.T_Cam_BodyImu_at(i);
    XR_CHECK_GE(thisVar.sensorIndex, 0);
    if (thisVar.prevVarIndex < 0) {
      continue;
    }
    const auto& prevVar = prob_.T_Cam_BodyImu_at(thisVar.prevVarIndex);

    VecX diagCov = camExtrRandomWalkCov(
        /* dtSec = */ (thisVar.averageTimestampUs - prevVar.averageTimestampUs) * 1e-6
        /* deltaTempC = */);
    diagCov *= camExtrRWinflate;
    VecX diagSqrtH = diagCov.cwiseInverse().cwiseSqrt();
    prob_.addCamExtrRWFactor(thisVar.prevVarIndex, i, diagSqrtH);
    nAdded++;
  }
  if (verbosity_ != Muted) {
    XR_LOGI("Camera RW on imu extrinsics: {}", nAdded);
  }
}

} // namespace visual_inertial_ba
