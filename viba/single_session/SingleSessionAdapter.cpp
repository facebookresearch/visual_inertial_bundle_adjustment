/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
#include <viba/common/Enumerate.h>
#include <viba/common/Settings.h>
#include <viba/problem/Histograms.h>
#include <viba/single_session/SingleSessionAdapter.h>

#include <logging/Checks.h>
#define DEFAULT_LOG_CHANNEL "ViBa::SingleSessionAdapter"
#include <logging/Log.h>

namespace visual_inertial_ba {

SingleSessionAdapter::SingleSessionAdapter(
    SingleSessionProblem& prob,
    const SessionData& fData,
    const Matcher& matcher,
    Verbosity verbosity)
    : verbosity_(verbosity),
      prob_(prob),
      fData_(fData),
      matcher_(matcher),
      numRigsInRecording_(matcher_.rigIndexToEvolvingStateIndex.size()),
      numCameras_(fData_.slamInfo.cameraSerialNumbers.size()),
      numImus_(fData_.slamInfo.imuLabels.size()) {
  if (verbosity_ != Muted) {
    XR_LOGI("#IMUs: {}, #cameras: {}", numImus_, numCameras_);
  }
}

void SingleSessionAdapter::initNonRigParams(const ParamInitSettings& settings) {
  initCamIntrinsics(
      settings.camIntrConstant,
      settings.estimateReadoutTime,
      settings.estimateTimeOffset,
      settings.camIntrToFactory);

  initCamExtrinsics(settings.camExtrConstant, settings.camExtrToFactory);

  initImuCalibs(
      settings.imuCalibConstant,
      settings.imuCalibToFactory,
      settings.imuCalibEstimationOptions,
      settings.imuCalibSetToFactory);

  if (numImus_ > 1) {
    initImuExtrinsics(settings.imuExtrConstant, settings.imuExtrToFactory);
  }

  // note: we need to update rolling shutter data
  // - after intrinsics are initialized
  // - before points are triangulated
  updateRollingShutterData();

  if (verbosity_ != Muted) {
    XR_LOGI("Initializing tracking points from observation (re-triangulating)...");
  }
  initPointsFromObservations();
}

void SingleSessionAdapter::initAllVariablesAndFactors(
    const InitSettings& s,
    int64_t rigIndexStart,
    int64_t rigIndexEnd,
    const TrajectoryBase* trajectory) {
  ParamInitSettings paramInitSettings{
      .rigIndexStart = rigIndexStart,
      .rigIndexEnd = rigIndexEnd,
  };
  paramInitSettings.setTrajectoryConstants(s.trajectoryConstant);
  paramInitSettings.setTrajectoryToGt(s.trajectoryToGt);
  paramInitSettings.setCalibConstants(s.calibConstant);
  paramInitSettings.setCalibToFactory(s.calibFactory);
  paramInitSettings.setImuCalibEstimationOptions(s.imuCalibEstimationOptions);
  paramInitSettings.setImuCalibToFactory(s.imuCalibFieldsToFactory);
  paramInitSettings.estimateReadoutTime = s.estimateReadoutTime;
  paramInitSettings.estimateTimeOffset = s.estimateTimeOffset;
  if (verbosity_ != Muted) {
    paramInitSettings.printImuCalibOptions();
  }

  if (!paramInitSettings.poseToGt && !paramInitSettings.velToGt && !paramInitSettings.omegaToGt) {
    initAllParams(paramInitSettings);
  } else {
    initAllParamsWithGtTrajectory(paramInitSettings, trajectory);
  }

  addVisualFactors(s.trackingObsLossRadius, s.trackingObsLossCutoff, s.optimizeDetectorBias);

  // required prior to adding inertial factors and omega priors
  regenerateAllPreintegrationsFromImuMeasurements();

  addInertialFactors(s.imuLossRadius, s.imuLossCutoff);

  if (verbosity_ != Muted) {
    XR_LOGI("Adding all random walk factors");
  }
  addAllRandomWalkFactors(
      s.imuRWinflate, s.camIntrRWinflate, s.imuExtrRWinflate, s.camExtrRWinflate);

  if (verbosity_ != Muted) {
    XR_LOGI("Adding omega prior factors");
  }
  addOmegaPriors();

  // factory calibration priors
  if (s.imuFactorCalibInflate > 0.0) {
    addImuFactoryCalibPriors(s.imuFactorCalibInflate);
  }

  if (s.camIntrFactorCalibInflate > 0.0) {
    addCamIntrFactoryCalibPriors(s.camIntrFactorCalibInflate);
  }

  if (s.imuExtrFactorCalibInflate > 0.0) {
    addImuExtrFactoryCalibPriors(s.imuExtrFactorCalibInflate);
  }

  if (s.camExtrFactorCalibInflate > 0.0) {
    addCamExtrFactoryCalibPriors(s.camExtrFactorCalibInflate);
  }
}

void SingleSessionAdapter::initAllParams(const ParamInitSettings& settings) {
  // validate settings:
  constexpr int kMinNumberRigs = 5;
  const int64_t rigIndexStart = settings.rigIndexStart >= 0 ? settings.rigIndexStart : 0;
  const int64_t rigIndexEnd =
      settings.rigIndexEnd >= 0 ? settings.rigIndexEnd : numRigsInRecording_;

  XR_CHECK_GE(
      rigIndexEnd - rigIndexStart,
      kMinNumberRigs,
      "Too small problem size, requested rig range {}..{}",
      rigIndexStart,
      rigIndexEnd);

  prob_.gravityWorld_initNew({
      .radius = kDefaultGravityMagnitude,
      .vec = Vec3{0, 0, -kDefaultGravityMagnitude},
  });

  if (verbosity_ != Muted) {
    XR_LOGI("Loading rigs...");
  }
  initRigs(rigIndexStart, rigIndexEnd);

  if (settings.poseConstant || settings.velConstant || settings.omegaConstant) {
    prob_.setAllRigsConstant(settings.poseConstant, settings.velConstant, settings.omegaConstant);
  }

  initNonRigParams(settings);
}

void SingleSessionAdapter::initAllParamsInterpolatingRigPoses(
    const ParamInitSettings& settings,
    const std::vector<TimeStampInterval>& timeIntervals,
    const std::vector<int64_t>& krTimestampsUs,
    const std::vector<Sophus::SE3d>& Ts_kr_world,
    const TrajectoryBase* trajectory,
    const std::shared_ptr<GravityVariable>& gravityWorld,
    StatsValueContainer* frameDistortionStats_relRot,
    StatsValueContainer* frameDistortionStats_relTr,
    int rigWindowGrow) {
  prob_.gravityWorld_setRef(gravityWorld);

  if (verbosity_ != Muted) {
    XR_LOGI("Loading rigs with map priors...");
  }
  initRigsInterpolatingPoses(
      timeIntervals,
      krTimestampsUs,
      Ts_kr_world,
      trajectory,
      settings.velToGt,
      settings.omegaToGt,
      frameDistortionStats_relRot,
      frameDistortionStats_relTr,
      rigWindowGrow);

  if (settings.poseConstant || settings.velConstant || settings.omegaConstant) {
    prob_.setAllRigsConstant(settings.poseConstant, settings.velConstant, settings.omegaConstant);
  }

  initNonRigParams(settings);
}

void SingleSessionAdapter::initAllParamsWithGtTrajectory(
    const ParamInitSettings& settings,
    const std::vector<TimeStampInterval>& timeIntervals,
    const std::vector<int64_t>& krTimestampsUs,
    const TrajectoryBase* trajectory,
    const std::shared_ptr<GravityVariable>& gravityWorld,
    int rigWindowGrow) {
  prob_.gravityWorld_setRef(gravityWorld);

  if (verbosity_ != Muted) {
    XR_LOGI("Loading rigs with gt poses...");
  }
  initRigsFromGtTrajectory(
      timeIntervals,
      krTimestampsUs,
      trajectory,
      settings.poseToGt,
      settings.velToGt,
      settings.omegaToGt,
      rigWindowGrow);

  if (settings.poseConstant || settings.velConstant || settings.omegaConstant) {
    prob_.setAllRigsConstant(settings.poseConstant, settings.velConstant, settings.omegaConstant);
  }

  initNonRigParams(settings);
}

void SingleSessionAdapter::initAllParamsWithGtTrajectory(
    const ParamInitSettings& settings,
    const TrajectoryBase* trajectory) {
  // validate settings:
  constexpr int kMinNumberRigs = 5;
  const int64_t rigIndexStart = settings.rigIndexStart >= 0 ? settings.rigIndexStart : 0;
  const int64_t rigIndexEnd =
      settings.rigIndexEnd >= 0 ? settings.rigIndexEnd : numRigsInRecording_;

  XR_CHECK_GE(
      rigIndexEnd - rigIndexStart,
      kMinNumberRigs,
      "Too small problem size, requested rig range {}..{}",
      rigIndexStart,
      rigIndexEnd);

  // gravity from GT trajectory if and only if we use GT poses
  prob_.gravityWorld_initNew({
      .radius = kDefaultGravityMagnitude,
      .vec = settings.poseToGt ? trajectory->gravity() : Vec3{0, 0, -kDefaultGravityMagnitude},
  });

  if (verbosity_ != Muted) {
    XR_LOGI("Loading all rigs with gt poses...");
  }
  initRigsFromGtTrajectory(
      rigIndexStart,
      rigIndexEnd,
      trajectory,
      settings.poseToGt,
      settings.velToGt,
      settings.omegaToGt);

  if (settings.poseConstant || settings.velConstant || settings.omegaConstant) {
    prob_.setAllRigsConstant(settings.poseConstant, settings.velConstant, settings.omegaConstant);
  }

  initNonRigParams(settings);
}

int64_t SingleSessionAdapter::usedRecordingLengthUs() const {
  auto rigs = prob_.sortedRigIndices();
  const auto& iPoseStart = fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[rigs[0]]];
  const auto& iPoseEnd = fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[rigs.back()]];
  return iPoseEnd.timestamp_us - iPoseStart.timestamp_us;
}

SingleSessionProblem& SingleSessionAdapter::getProblem() {
  return prob_;
}

const SingleSessionProblem& SingleSessionAdapter::getProblem() const {
  return prob_;
}

} // namespace visual_inertial_ba
