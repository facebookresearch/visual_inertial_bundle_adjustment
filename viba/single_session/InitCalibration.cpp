/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/single_session/SingleSessionAdapter.h>

#include <logging/Checks.h>
#define DEFAULT_LOG_CHANNEL "ViBa::InitCalibration"
#include <logging/Log.h>

namespace visual_inertial_ba {

template <typename F>
static void eachToken(const std::string& args, F&& f) {
  std::istringstream ss(args);
  std::string token;

  while (std::getline(ss, token, ',').good()) {
    const bool val = token[0] != '-';
    const int start = (token[0] == '-') ? 1 : 0;
    f(token.substr(start), val);
  }

  ss >> token;
  if (!token.empty()) {
    const bool val = token[0] != '-';
    const int start = (token[0] == '-') ? 1 : 0;
    f(token.substr(start), val);
  }
}

static void parseCalibOptionString(ImuCalibrationOptions& estOpts, const std::string& args) {
  eachToken(args, [&](const std::string& token, bool val) {
    if (token == "gyro-bias") {
      estOpts.gyroBias = val;
    } else if (token == "accel-bias") {
      estOpts.accelBias = val;
    } else if (token == "gyro-scale") {
      estOpts.gyroScale = val;
    } else if (token == "accel-scale") {
      estOpts.accelScale = val;
    } else if (token == "gyro-nonorth") {
      estOpts.gyroNonOrth = val;
    } else if (token == "accel-nonorth") {
      estOpts.accelNonOrth = val;
    } else if (token == "reference-imu-time-offset") {
      estOpts.referenceImuTimeOffset = val;
    } else if (token == "gyro-accel-time-offset") {
      estOpts.gyroAccelTimeOffset = val;
    } else if (token == "all") {
      estOpts.gyroBias = estOpts.accelBias = estOpts.gyroScale = estOpts.accelScale =
          estOpts.gyroNonOrth = estOpts.accelNonOrth = estOpts.referenceImuTimeOffset =
              estOpts.gyroAccelTimeOffset = val;
    } else if (token == "all-but-time-offsets") {
      estOpts.gyroBias = estOpts.accelBias = estOpts.gyroScale = estOpts.accelScale =
          estOpts.gyroNonOrth = estOpts.accelNonOrth = val;
    } else if (token == "all-but-biases") {
      estOpts.gyroScale = estOpts.accelScale = estOpts.gyroNonOrth = estOpts.accelNonOrth =
          estOpts.referenceImuTimeOffset = estOpts.gyroAccelTimeOffset = val;
    } else if (token == "all-but-biases-and-scales") {
      estOpts.gyroNonOrth = estOpts.accelNonOrth = estOpts.referenceImuTimeOffset =
          estOpts.gyroAccelTimeOffset = val;
    } else if (token == "all-non-orths") {
      estOpts.referenceImuTimeOffset = estOpts.gyroAccelTimeOffset = val;
    } else if (token == "all-time-offsets") {
      estOpts.gyroNonOrth = estOpts.accelNonOrth = val;
    }
  });
}

static void setInertialPoseFromArgs(bool* pose, bool* vel, bool* omega, const std::string& args) {
  eachToken(args, [&](const std::string& token, bool val) {
    if (token == "pose") {
      *pose = val;
    } else if (token == "vel") {
      *vel = val;
    } else if (token == "omega") {
      *omega = val;
    } else if (token == "all") {
      *pose = *vel = *omega = val;
    } else {
      XR_CHECK(false, "Unknown inertial-pose spec token: '{}'", token);
    }
  });
}

void SingleSessionAdapter::ParamInitSettings::setTrajectoryConstants(const std::string& args) {
  setInertialPoseFromArgs(&poseConstant, &velConstant, &poseConstant, args);
}

void SingleSessionAdapter::ParamInitSettings::setTrajectoryToGt(const std::string& args) {
  setInertialPoseFromArgs(&poseToGt, &velToGt, &poseToGt, args);
}

static void setCalibFromArgs(
    bool* camIntr,
    bool* camExtr,
    bool* imuCalib,
    bool* imuExtr,
    const std::string& args) {
  eachToken(args, [&](const std::string& token, bool val) {
    if (token == "imu-calib") {
      *imuCalib = val;
    } else if (token == "imu-extr") {
      *imuExtr = val;
    } else if (token == "imu-all") {
      *imuCalib = *imuExtr = val;
    } else if (token == "cam-intr") {
      *camIntr = val;
    } else if (token == "cam-extr") {
      *camExtr = val;
    } else if (token == "cam-all") {
      *camIntr = *camExtr = val;
    } else if (token == "all-extr") {
      *camExtr = *imuExtr = val;
    } else if (token == "all") {
      *camIntr = *camExtr = *imuCalib = *imuExtr = val;
    } else {
      XR_CHECK(false, "Unknown calib spec token: '{}'", token);
    }
  });
}

void SingleSessionAdapter::ParamInitSettings::setCalibConstants(const std::string& args) {
  setCalibFromArgs(&camIntrConstant, &camExtrConstant, &imuCalibConstant, &imuExtrConstant, args);
}

void SingleSessionAdapter::ParamInitSettings::setCalibToFactory(const std::string& args) {
  setCalibFromArgs(
      &camIntrToFactory, &camExtrToFactory, &imuCalibToFactory, &imuExtrToFactory, args);
}

void SingleSessionAdapter::ParamInitSettings::setImuCalibEstimationOptions(
    const std::string& args) {
  imuCalibEstimationOptions = kDefaultImuCalibEstimationOptions;
  parseCalibOptionString(imuCalibEstimationOptions, args);
}

void SingleSessionAdapter::ParamInitSettings::setImuCalibToFactory(const std::string& args) {
  imuCalibSetToFactory = ImuCalibrationOptions(false);
  parseCalibOptionString(imuCalibSetToFactory, args);
}

void SingleSessionAdapter::ParamInitSettings::printImuCalibOptions() const {
  printEstOptionsVsDefaults(
      fmt::format(
          "Imu calibration estimation options (error state size: {})",
          imuCalibEstimationOptions.errorStateSize()),
      imuCalibEstimationOptions,
      kDefaultImuCalibEstimationOptions);
  if (imuCalibSetToFactory.errorStateSize() > 0) {
    printEstOptionsVsDefaults(
        "Imu calibration fields init to factory calibration",
        imuCalibSetToFactory,
        ImuCalibrationOptions(false));
  }
}

constexpr double kImuCalibGroupTimeLengthSec = 5.0;

constexpr double kCameraModelGroupTimeLengthSec = 5.0;

constexpr double kExtrinsicsGroupTimeLengthSec = 5.0;

// TODO: this could be moved inside SingleSessionProblem class
std::vector<int64_t> SingleSessionAdapter::rigWindowsOfTimeLengthAtMost(double maxTimeLengthSec) {
  const int64_t maxTimeLengthUs = int64_t(maxTimeLengthSec * 1e6);
  const auto allRigIndices = prob_.sortedRigIndices();
  std::vector<int64_t> retv;
  int64_t startCurrentWindow = prob_.inertialPose(allRigIndices[0]).timestampUs - maxTimeLengthUs;
  for (const auto rigIndex : allRigIndices) {
    const int64_t timestampUs = prob_.inertialPose(rigIndex).timestampUs;
    if (timestampUs - startCurrentWindow >= maxTimeLengthUs) {
      startCurrentWindow = timestampUs;
      retv.push_back(rigIndex);
    }
  }
  retv.push_back(allRigIndices.back() + 1);
  return retv;
}

template <typename T, typename F>
static std::optional<int64_t> last_in_range(const T& start, const T& end, F&& func) {
  for (T i = end; i > start; i--) {
    if (func(i - 1)) {
      return i - 1;
    }
  }
  return std::nullopt;
}

void SingleSessionAdapter::initCamIntrinsics(
    bool constant,
    bool estimateReadoutTime,
    bool estimateTimeOffset,
    bool useFactory) {
  XR_CHECK_GT(prob_.inertialPoses().size(), 0);

  if (useFactory) {
    if (verbosity_ != Muted) {
      XR_LOGI("Loading camera intrinsics, constant GT values from VRS...");
    }

    for (int s = 0; s < numCameras_; s++) {
      const int fIdx = matcher_.slamCamIndexToFactoryCalib[s];
      auto factoryModelParams = fData_.factoryCalibration.calib.getConvertedCameraModelParam(fIdx);
      factoryModelParams.setEstimateReadoutTime(false);
      factoryModelParams.setEstimateTimeOffset(false);

      const int newVarIndex = prob_.cameraModel_addNew(
          std::move(factoryModelParams),
          /* averageTimestampUs = */ 0,
          /* prevVarIndex = */ -1,
          /* sensorIndex = */ s,
          /* constant = */ true);

      for (const auto& [j, _] : prob_.inertialPoses()) {
        prob_.cameraModel_setRef(j, s, newVarIndex);
      }
    }

    return;
  }

  const auto groupCalibRanges = rigWindowsOfTimeLengthAtMost(kCameraModelGroupTimeLengthSec); //
  if (verbosity_ != Muted) {
    XR_LOGI(
        "Loading camera intrinsics (opt {}) from online calibration data, grouped into {} windows of ~{:.01f}s",
        constant ? "constants" : "variables",
        groupCalibRanges.size() - 1,
        kCameraModelGroupTimeLengthSec);
  }

  // collect larget readout time for each rig
  const double kTimestampSlackSec = 1e-3; // 1 ms
  std::unordered_map<int64_t, double> rigToMaxCameraReadoutTimeSec;

  for (int s = 0; s < numCameras_; s++) {
    const int olIdx = matcher_.slamCamIndexToOnlineCalib.at(s);
    XR_CHECK_GE(olIdx, 0);

    int64_t prevVarIndex = -1;
    for (int i = 0; i < groupCalibRanges.size() - 1; i++) {
      const int64_t start = groupCalibRanges[i];
      const int64_t end = groupCalibRanges[i + 1];
      const auto maybeLastInProblem =
          last_in_range(start, end, [&](int64_t j) { return prob_.inertialPose_exists(j); });
      const int64_t averageTimestampUs = prob_.averageTimestampOfRigsInRange(start, end);
      if (maybeLastInProblem.has_value()) {
        // let's grab the calib state at the end of the window
        const int calibStateIndex = matcher_.rigIndexToCalibStateIndex[*maybeLastInProblem];
        const auto& calibState = fData_.onlineCalibration.calibs[calibStateIndex];
        auto onlineParams = calibState.getConvertedCameraModelParam(olIdx);

        // ATM time offset and readout time estimation, when enabled, will be enabled only for
        // rolling shutter cameras.
        const bool isRollingShutter = onlineParams.isRollingShutter();
        const double timeOffsetsSeconds_Ref_Cam = onlineParams.timeOffsetSec_Dev_Camera();
        const bool thisCamEstimateReadoutTime = estimateReadoutTime && isRollingShutter;
        const bool thisCamEstimateTimeOffset = estimateTimeOffset && isRollingShutter;

        // we want an integration interval, if either we enable estimation, either we need it
        // to correctly compute the factor from (possibly constant) readout/offset timings
        const double camTimeSpan =
            (isRollingShutter ? onlineParams.readoutTimeSec() + kTimestampSlackSec : 0.0) +
            ((thisCamEstimateTimeOffset || (timeOffsetsSeconds_Ref_Cam != 0.0))
                 ? 2.0 * (std::abs(timeOffsetsSeconds_Ref_Cam) + kTimestampSlackSec)
                 : 0.0);
        XR_CHECK_LT(camTimeSpan, 1.0);

        onlineParams.setEstimateReadoutTime(thisCamEstimateReadoutTime);
        onlineParams.setEstimateTimeOffset(thisCamEstimateTimeOffset);

        const int64_t newVarIndex = prob_.cameraModel_addNew(
            std::move(onlineParams), averageTimestampUs, prevVarIndex, s, constant);
        prevVarIndex = newVarIndex;

        for (int64_t j = start; j < end; j++) {
          if (prob_.inertialPose_exists(j)) {
            prob_.cameraModel_setRef(j, s, newVarIndex);
            if (camTimeSpan > 0) {
              double& curVal = rigToMaxCameraReadoutTimeSec[j];
              curVal = std::max(curVal, camTimeSpan);
            }
          }
        }
      }
    }
  }

  if (!rigToMaxCameraReadoutTimeSec.empty()) {
    initRollingShutterData(rigToMaxCameraReadoutTimeSec, fData_.allImuMeasurements[0]);
  }
}

void SingleSessionAdapter::initRollingShutterData(
    const std::unordered_map<int64_t, double>& rigToMaxCameraReadoutTimeSec,
    const std::vector<ImuMeasurement>& imu0Measurements) {
  const int64_t kTimestampSlackMs = 2;

  for (const auto& [j, roTimeSec] : rigToMaxCameraReadoutTimeSec) {
    const int calibStateIndex = matcher_.rigIndexToCalibStateIndex[j];
    const int64_t timestampUs = fData_.onlineCalibration.calibs[calibStateIndex].timestamp_us;
    const int64_t intervalHalfSizeUs = kTimestampSlackMs * 1e3 + roTimeSec * 0.5e6;

    prob_.rigToRSData().emplace(
        std::piecewise_construct,
        std::forward_as_tuple(j),
        std::forward_as_tuple(timestampUs, intervalHalfSizeUs, imu0Measurements));
  }
}

void SingleSessionAdapter::updateRollingShutterData() {
  Vec3 gravityWorld = prob_.gravityWorld().value.vec;
  for (auto& [j, rsData] : prob_.rigToRSData()) {
    const int calibStateIndex = matcher_.rigIndexToCalibStateIndex[j];
    const int64_t timestampUs = fData_.onlineCalibration.calibs[calibStateIndex].timestamp_us;

    const auto& imuCalib = prob_.imuCalib(j, 0); // imu 0 data
    rsData.compute(timestampUs, imuCalib.var.value.modelParams, gravityWorld);
  }
}

void SingleSessionAdapter::initCamExtrinsics(bool constant, bool useFactory) {
  XR_CHECK_GT(prob_.inertialPoses().size(), 0);

  if (useFactory) {
    if (verbosity_ != Muted) {
      XR_LOGI("Loading camera extrinsics, constant GT values from VRS...");
    }

    for (int s = 0; s < numCameras_; s++) {
      const int fIdx = matcher_.slamCamIndexToFactoryCalib[s];
      const SE3& factoryT_Cam_BodyImu = fData_.factoryCalibration.calib.T_Cam_BodyImu[fIdx];
      const int newVarIndex = prob_.T_Cam_BodyImu_addNew(
          factoryT_Cam_BodyImu,
          /* averageTimestampUs = */ 0,
          /* prevVarIndex = */ -1,
          /* sensorIndex = */ s,
          /* constant = */ true);
      for (const auto& [j, _] : prob_.inertialPoses()) {
        prob_.T_Cam_BodyImu_setRef(j, s, newVarIndex);
      }
    }

    return;
  }

  const auto groupCalibRanges = rigWindowsOfTimeLengthAtMost(kExtrinsicsGroupTimeLengthSec); //
  if (verbosity_ != Muted) {
    XR_LOGI(
        "Loading camera extrinsics (opt {}) from online calibration data, grouped into {} windows of ~{:.01f}s",
        constant ? "constants" : "variables",
        groupCalibRanges.size() - 1,
        kCameraModelGroupTimeLengthSec);
  }

  for (int s = 0; s < numCameras_; s++) {
    const int olIdx = matcher_.slamCamIndexToOnlineCalib.at(s);

    int64_t prevVarIndex = -1;
    for (int i = 0; i < groupCalibRanges.size() - 1; i++) {
      const int64_t start = groupCalibRanges[i];
      const int64_t end = groupCalibRanges[i + 1];
      const auto maybeLastInProblem =
          last_in_range(start, end, [&](int64_t j) { return prob_.inertialPose_exists(j); });
      const int64_t averageTimestampUs = prob_.averageTimestampOfRigsInRange(start, end);
      if (maybeLastInProblem.has_value()) {
        // let's grab the extrinsics at the end of the window
        const int calibStateIndex = matcher_.rigIndexToCalibStateIndex[*maybeLastInProblem];
        const auto& calibState = fData_.onlineCalibration.calibs[calibStateIndex];
        const int newVarIndex = prob_.T_Cam_BodyImu_addNew(
            calibState.T_Cam_BodyImu[olIdx], averageTimestampUs, prevVarIndex, s, constant);
        prevVarIndex = newVarIndex;

        for (int64_t j = start; j < end; j++) {
          if (prob_.inertialPose_exists(j)) {
            prob_.T_Cam_BodyImu_setRef(j, s, newVarIndex);
          }
        }
      }
    }
  }
}

static void selectivelySetImuCalibFields(
    ImuMeasurementModelParameters& imuModel,
    const ImuMeasurementModelParameters& altImuModel,
    const ImuCalibrationOptions& s) {
  if (s.gyroBias) {
    imuModel.gyroBiasRadSec = altImuModel.gyroBiasRadSec;
  }
  if (s.accelBias) {
    imuModel.accelBiasMSec2 = altImuModel.accelBiasMSec2;
  }
  if (s.gyroScale) {
    imuModel.gyroScaleVec = altImuModel.gyroScaleVec;
  }
  if (s.accelScale) {
    imuModel.accelScaleVec = altImuModel.accelScaleVec;
  }
  if (s.gyroNonOrth) {
    imuModel.gyroNonorth = altImuModel.gyroNonorth;
  }
  if (s.accelNonOrth) {
    imuModel.accelNonorth = altImuModel.accelNonorth;
  }
  if (s.referenceImuTimeOffset) { // preserve estimate of (dtAccel - dtGyro)
    imuModel.dtReferenceAccelSec += altImuModel.dtReferenceGyroSec - imuModel.dtReferenceGyroSec;
    imuModel.dtReferenceGyroSec = altImuModel.dtReferenceGyroSec;
  }
  if (s.gyroAccelTimeOffset) { // make (dtAccel - dtGyro) become like in GT
    imuModel.dtReferenceAccelSec =
        (altImuModel.dtReferenceAccelSec - altImuModel.dtReferenceGyroSec) +
        imuModel.dtReferenceGyroSec;
  }
}

void SingleSessionAdapter::initImuCalibs(
    bool constant,
    bool useFactory,
    const ImuCalibrationOptions& argImuCalibEstimationOptions,
    const ImuCalibrationOptions& argImuCalibSetToFactory) {
  XR_CHECK_GT(prob_.inertialPoses().size(), 0);

  // init estimation options
  imuCalibEstimationOptions = argImuCalibEstimationOptions;
  imuCalibJacobianIndices.computeIndices(imuCalibEstimationOptions);

  if (useFactory) {
    if (verbosity_ != Muted) {
      XR_LOGI("Loading imu calibs, constant GT values from VRS...");
    }

    for (int s = 0; s < numImus_; s++) {
      const int fIdx = matcher_.slamImuIndexToFactoryCalib[s];
      const auto& factoryModelParams = fData_.factoryCalibration.calib.imuModelParameters[fIdx];

      const int newVarIndex = prob_.imuCalib_addNew(
          ImuCalibParam{
              .modelParams = factoryModelParams,
              .estOpts = &imuCalibEstimationOptions,
              .jacInd = &imuCalibJacobianIndices,
          },
          /* averageTimestampUs = */ 0,
          /* prevVarIndex = */ -1,
          /* sensorIndex = */ s,
          /* constant = */ true);
      for (const auto& [j, _] : prob_.inertialPoses()) {
        prob_.imuCalib_setRef(j, s, newVarIndex);
      }
    }

    return;
  }

  const bool anyFromFactoryCalib = argImuCalibSetToFactory.errorStateSize() > 0;

  const auto groupCalibRanges = rigWindowsOfTimeLengthAtMost(kImuCalibGroupTimeLengthSec); //
  if (verbosity_ != Muted) {
    XR_LOGI(
        "Loading imu calibs (opt {}) from online calibration data, grouped into {} windows of ~{:.01f}s",
        constant ? "constants" : "variables",
        groupCalibRanges.size() - 1,
        kCameraModelGroupTimeLengthSec);
  }

  for (int s = 0; s < numImus_; s++) {
    const int olIdx = matcher_.slamImuIndexToOnlineCalib.at(s);

    int64_t prevVarIndex = -1;
    for (int i = 0; i < groupCalibRanges.size() - 1; i++) {
      const int64_t start = groupCalibRanges[i];
      const int64_t end = groupCalibRanges[i + 1];
      const auto maybeLastInProblem =
          last_in_range(start, end, [&](int64_t j) { return prob_.inertialPose_exists(j); });
      const int64_t averageTimestampUs = prob_.averageTimestampOfRigsInRange(start, end);
      if (maybeLastInProblem.has_value()) {
        // let's grab the extrinsics at the end of the window
        const int calibStateIndex = matcher_.rigIndexToCalibStateIndex[*maybeLastInProblem];
        const auto& calibState = fData_.onlineCalibration.calibs[calibStateIndex];

        // selectively set, if needed
        ImuMeasurementModelParameters modelParams = calibState.imuModelParameters[olIdx];
        if (anyFromFactoryCalib) {
          const int fIdx = matcher_.slamImuIndexToFactoryCalib[s];
          const auto& factoryModelParams = fData_.factoryCalibration.calib.imuModelParameters[fIdx];

          selectivelySetImuCalibFields(modelParams, factoryModelParams, argImuCalibSetToFactory);
        }

        const int64_t newVarIndex = prob_.imuCalib_addNew(
            ImuCalibParam{
                .modelParams = modelParams,
                .estOpts = &imuCalibEstimationOptions,
                .jacInd = &imuCalibJacobianIndices,
            },
            averageTimestampUs,
            prevVarIndex,
            s,
            constant);
        prevVarIndex = newVarIndex;

        for (int64_t j = start; j < end; j++) {
          if (prob_.inertialPose_exists(j)) {
            prob_.imuCalib_setRef(j, s, newVarIndex);
          }
        }
      }
    }
  }
}

void SingleSessionAdapter::initImuExtrinsics(bool constant, bool useFactory) {
  XR_CHECK_GT(prob_.inertialPoses().size(), 0);

  if (useFactory) {
    if (verbosity_ != Muted) {
      XR_LOGI("Loading imu extrinsics, constant GT values from VRS...");
    }

    for (int s = 0; s < numImus_; s++) {
      const int fIdx = matcher_.slamImuIndexToFactoryCalib[s];
      const SE3& factoryT_Imu_BodyImu = fData_.factoryCalibration.calib.T_Imu_BodyImu[fIdx];
      const int newVarIndex = prob_.T_Imu_BodyImu_addNew(
          factoryT_Imu_BodyImu,
          /* averageTimestampUs = */ 0,
          /* prevVarIndex = */ -1,
          /* sensorIndex = */ s,
          /* constant = */ true);
      for (const auto& [j, _] : prob_.inertialPoses()) {
        prob_.T_Imu_BodyImu_setRef(j, s, newVarIndex);
      }
    }

    return;
  }

  const auto groupCalibRanges = rigWindowsOfTimeLengthAtMost(kExtrinsicsGroupTimeLengthSec); //
  if (verbosity_ != Muted) {
    XR_LOGI(
        "Loading imu extrinsics (opt {}) from online calibration data, grouped into {} windows of ~{:.01f}s",
        constant ? "constants" : "variables",
        groupCalibRanges.size() - 1,
        kCameraModelGroupTimeLengthSec);
  }

  // start from 1, skipping imu0 which is taken to have trivial extrinsics
  for (int s = 1; s < numImus_; s++) {
    const int olIdx = matcher_.slamImuIndexToOnlineCalib.at(s);

    int64_t prevVarIndex = -1;
    for (int i = 0; i < groupCalibRanges.size() - 1; i++) {
      const int64_t start = groupCalibRanges[i];
      const int64_t end = groupCalibRanges[i + 1];
      const auto maybeLastInProblem =
          last_in_range(start, end, [&](int64_t j) { return prob_.inertialPose_exists(j); });
      const int64_t averageTimestampUs = prob_.averageTimestampOfRigsInRange(start, end);
      if (maybeLastInProblem.has_value()) {
        // let's grab the extrinsics at the end of the window
        const int calibStateIndex = matcher_.rigIndexToCalibStateIndex[*maybeLastInProblem];
        const auto& calibState = fData_.onlineCalibration.calibs[calibStateIndex];

        const int64_t newVarIndex = prob_.T_Imu_BodyImu_addNew(
            calibState.T_Imu_BodyImu[olIdx], averageTimestampUs, prevVarIndex, s, constant);
        prevVarIndex = newVarIndex;

        for (int64_t j = start; j < end; j++) {
          if (prob_.inertialPose_exists(j)) {
            prob_.T_Imu_BodyImu_setRef(j, s, newVarIndex);
          }
        }
      }
    }
  }
}

} // namespace visual_inertial_ba
