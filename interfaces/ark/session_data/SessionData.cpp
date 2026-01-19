/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <session_data/SessionData.h>

#include <imu_types/ImuDataReader.h>
#include <nlohmann/json.hpp>
#include <point_observation/PointObservationReader.h>
#include <projectaria_tools/core/calibration/loader/DeviceCalibrationJson.h>
#include <projectaria_tools/core/mps/OnlineCalibrationsReader.h>
#include <projectaria_tools/core/mps/TrajectoryReaders.h>
#include <session_data/ImuCalibConversion.h>
#include <iostream>

#define DEFAULT_LOG_CHANNEL "ViBa::SessionData"
#include <logging/Checks.h>
#include <logging/Log.h>

namespace visual_inertial_ba {

using namespace projectaria::tools;

#define USE_OPEN_LOOP 1

constexpr const char* kVrsSourceInfo = "vrs_source_info.json";
constexpr const char* kOnlineCalibration = "online_calibration.jsonl";
constexpr const char* kFactoryCalibration = "factory_calibration.json";
constexpr const char* kPointObservations = "session_observations.csv";
constexpr const char* kImuSamplesPattern = "imu_samples_{}.csv";
#if USE_OPEN_LOOP
constexpr const char* kOpenLoopTrajectory = "open_loop_trajectory.csv";
#else
constexpr const char* kClosedLoopTrajectory = "closed_loop_framerate_trajectory.csv";
#endif

template <typename T, typename F>
static std::string showElements(const std::vector<T>& data, int nStart, int nEnd, F&& f) {
  std::stringstream ss;
  ss << "[";
  for (int i = 0; i < std::min(nStart, (int)data.size()); i++) {
    ss << ((i % 5 == 0) ? "\n  " : "") << f(data[i]) << ", ";
  }
  ss << "\n  ...";
  for (int istart = std::max((int)data.size() - nEnd, 0), i = istart; i < (int)data.size(); i++) {
    ss << (((i - istart) % 5 == 0) ? "\n  " : "") << f(data[i])
       << ((i < (int)data.size() - 1) ? ", " : "");
  }
  ss << "\n]";
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const std::optional<double>& maybeVal) {
  if (maybeVal.has_value()) {
    os << *maybeVal;
  } else {
    os << "<none>";
  }
  return os;
}

std::string readWholeFile(const std::filesystem::path& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    throw std::runtime_error(fmt::format("File not found: {}", path.string()));
  }
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return buffer.str();
}

// CalibrationState class
CameraModelParam CalibrationState::getConvertedCameraModelParam(int cameraIndex) const {
  XR_CHECK_LT(cameraIndex, camParameters.size());
  return CameraModelParam(camParameters[cameraIndex], false, false);
}

void SessionData::load(
    const std::filesystem::path& sessDataPath,
    bool loadImuMeasurements,
    bool verbose) {
  // load slam info - also needed to know what imu is the bodyImu
  {
    std::ifstream ifs(sessDataPath / kVrsSourceInfo);
    if (!ifs) {
      throw std::runtime_error(
          fmt::format("File not found: {}", (sessDataPath / kVrsSourceInfo).string()));
    }
    auto json = nlohmann::json::parse(ifs);
    for (auto& elem : json["camera_ids"]) {
      slamInfo.cameraSerialNumbers.push_back(elem);
    }
    for (auto& elem : json["imu_ids"]) {
      slamInfo.imuLabels.push_back(elem);
    }

    for (size_t i = 0; i < slamInfo.cameraSerialNumbers.size(); i++) {
      std::cout << "SLAM camera n." << i << ": " << slamInfo.cameraSerialNumbers[i] << std::endl;
    }
    for (size_t i = 0; i < slamInfo.imuLabels.size(); i++) {
      std::cout << "SLAM imu n." << i << ": " << slamInfo.imuLabels[i] << std::endl;
    }
  }

  auto onlineCalibs = mps::readOnlineCalibration(sessDataPath / kOnlineCalibration);
  if (onlineCalibs.empty()) {
    throw std::runtime_error("Unable to load online calib!");
  }
  std::unordered_map<std::string, int> onlineCameraLabelToIndex;
  std::unordered_map<std::string, int> onlineImuLabelToIndex;
  for (size_t i = 0; i < onlineCalibs[0].cameraCalibs.size(); i++) {
    const auto& calib = onlineCalibs[0].cameraCalibs[i];
    onlineCameraLabelToIndex[calib.getLabel()] = i;
    std::cout << "ONLINE camera n." << i << ": " << calib.getLabel() << std::endl;
    std::cout << "  serial: " << calib.getSerialNumber() << std::endl;
    std::cout << "  image size: " << calib.getImageSize().x() << " x " << calib.getImageSize().y()
              << std::endl;
    std::cout << "  radius: " << calib.getValidRadius() << std::endl;
    std::cout << "  t-offset: " << calib.getTimeOffsetSecDeviceCamera() << std::endl;
    std::cout << "  readout-t: " << calib.getReadOutTimeSec() << std::endl;
  }
  for (size_t i = 0; i < onlineCalibs[0].imuCalibs.size(); i++) {
    const auto& calib = onlineCalibs[0].imuCalibs[i];
    onlineImuLabelToIndex[calib.getLabel()] = i;
    std::cout << "ONLINE imu n." << i << ": " << calib.getLabel() << std::endl;
  }

  // load factory calibration
  auto maybeFactoryCalib =
      deviceCalibrationFromJson(readWholeFile(sessDataPath / kFactoryCalibration));
  if (!maybeFactoryCalib.has_value()) {
    throw std::runtime_error("Unable to load factory calib!");
  }
  const auto& factoryCalib = *maybeFactoryCalib;
  {
    factoryCalibration.imuLabels = factoryCalib.getImuLabels();
    factoryCalibration.cameraLabels = factoryCalib.getCameraLabels();

    auto maybeT_device_bodyImu = factoryCalib.getT_Device_Sensor(slamInfo.imuLabels[0]);
    if (!maybeT_device_bodyImu) {
      throw std::runtime_error(
          fmt::format(
              "Slam's IMU n.0 = {} not present in factory calibration", slamInfo.imuLabels[0]));
    }
    slamInfo.T_bodyImu_device = maybeT_device_bodyImu->inverse();

    for (size_t i = 0; i < factoryCalibration.cameraLabels.size(); i++) {
      std::cout << "FACTORY camera n." << i << ": " << factoryCalibration.cameraLabels[i]
                << std::endl;
      auto maybeCalib = factoryCalib.getCameraCalib(factoryCalibration.cameraLabels[i]);
      if (!maybeCalib) {
        throw std::runtime_error("Camera not found");
      }
      auto& calib = *maybeCalib;
      std::cout << "  serial: " << calib.getSerialNumber() << std::endl;

      // If present in online calibration, adapt to online running resolution
      // Also copy radius, readout time and time offset parameters.
      auto onlineCalibIndexIt = onlineCameraLabelToIndex.find(factoryCalibration.cameraLabels[i]);
      if (onlineCalibIndexIt != onlineCameraLabelToIndex.end()) {
        const auto& oCalib = onlineCalibs[0].cameraCalibs[onlineCalibIndexIt->second];
        CameraCalibration adaptedCalib = calib;
        if (oCalib.getImageSize() != calib.getImageSize()) {
          std::cout << " !adapting image size: " << calib.getImageSize().x() << " x "
                    << calib.getImageSize().y() << " -> " << oCalib.getImageSize().x() << " x "
                    << oCalib.getImageSize().y() << std::endl;
          adaptedCalib = calib.rescale(
              oCalib.getImageSize(), double(oCalib.getImageSize().x()) / calib.getImageSize().x());
        } else {
          std::cout << "(not adapting, identical resolution found)" << std::endl;
        }
        calib = CameraCalibration(
            adaptedCalib.getLabel(),
            adaptedCalib.modelName(),
            adaptedCalib.projectionParams(),
            adaptedCalib.getT_Device_Camera(),
            adaptedCalib.getImageSize().x(),
            adaptedCalib.getImageSize().y(),
            oCalib.getValidRadius(),
            adaptedCalib.getMaxSolidAngle(),
            adaptedCalib.getSerialNumber(),
            oCalib.getTimeOffsetSecDeviceCamera(),
            oCalib.getReadOutTimeSec());
      }

      std::cout << "  image size: " << calib.getImageSize().x() << " x " << calib.getImageSize().y()
                << std::endl;
      std::cout << "  radius: " << calib.getValidRadius() << std::endl;
      std::cout << "  t-offset: " << calib.getTimeOffsetSecDeviceCamera() << std::endl;
      std::cout << "  readout-t: " << calib.getReadOutTimeSec() << std::endl;
      factoryCalibration.cameraSerialNumbers.push_back(calib.getSerialNumber());
      factoryCalibration.calib.camParameters.push_back(calib);
      factoryCalibration.calib.T_Cam_BodyImu.push_back(
          calib.getT_Device_Camera().inverse() * maybeT_device_bodyImu.value());
    }
    for (size_t i = 0; i < factoryCalibration.imuLabels.size(); i++) {
      std::cout << "FACTORY imu n." << i << ": " << factoryCalibration.imuLabels[i] << std::endl;
      auto maybeCalib = factoryCalib.getImuCalib(factoryCalibration.imuLabels[i]);
      if (!maybeCalib) {
        throw std::runtime_error("Imu not found");
      }
      auto& calib = *maybeCalib;
      factoryCalibration.calib.imuModelParameters.push_back(fromProjectAriaCalibration(calib));
      factoryCalibration.calib.T_Imu_BodyImu.push_back(
          calib.getT_Device_Imu().inverse() * maybeT_device_bodyImu.value());

      // setup noise - hard-code Aria values
      factoryCalibration.imuNoiseModels.emplace_back();
      if (factoryCalibration.imuLabels[i] == "imu-left") {
        factoryCalibration.imuNoiseModels.back().accelSampleVarianceM2Sec4.setConstant(
            7.7951241e-3);
        std::cout << "Hardcoded IMU noise model for: imu-left" << std::endl;
      } else if (factoryCalibration.imuLabels[i] == "imu-right") {
        factoryCalibration.imuNoiseModels.back().accelSampleVarianceM2Sec4.setConstant(
            6.6297049e-3);
        std::cout << "Hardcoded IMU noise model for: imu-right" << std::endl;
      } else {
        std::cout << "Unknown label: '" << factoryCalibration.imuLabels[i]
                  << "', using default IMU noise model" << std::endl;
      }
    }
  }

  // utility to show time stamps
  auto showTimeStamps = [](const auto& vec) {
    return showElements(vec, 10, 5, [&](auto el) { return el.timestamp_us; });
  };
  auto showTimeStamps3 = [](const auto& vec) {
    return showElements(vec, 10, 5, [&](auto el) { return el.captureTimestampUs; });
  };
  auto showTimeStamps4 = [](const auto& vec) {
    return showElements(vec, 10, 5, [&](auto el) { return el.timestampNs / 1000; });
  };

  // online calibration
  {
    for (const auto& onlineCalib : onlineCalibs) {
      std::vector<std::string> ocCameraSerialNumbers;
      std::vector<std::string> ocCameraLabels;
      std::vector<std::string> ocImuLabels;

      auto& loadedCalib = onlineCalibration.calibs.emplace_back();
      loadedCalib.timestamp_us = onlineCalib.trackingTimestamp.count();
      loadedCalib.camParameters = onlineCalib.cameraCalibs;
      for (const auto& camCalib : onlineCalib.cameraCalibs) {
        ocCameraSerialNumbers.push_back(camCalib.getSerialNumber());
        ocCameraLabels.push_back(camCalib.getLabel());
        loadedCalib.T_Cam_BodyImu.push_back(
            (slamInfo.T_bodyImu_device * camCalib.getT_Device_Camera()).inverse());
      }
      for (const auto& imuCalib : onlineCalib.imuCalibs) {
        ocImuLabels.push_back(imuCalib.getLabel());
        loadedCalib.imuModelParameters.push_back(fromProjectAriaCalibration(imuCalib));
        loadedCalib.T_Imu_BodyImu.push_back(
            (slamInfo.T_bodyImu_device * imuCalib.getT_Device_Imu()).inverse());
      }

      if (onlineCalibration.calibs.size() == 1) {
        onlineCalibration.cameraSerialNumbers = std::move(ocCameraSerialNumbers);
        onlineCalibration.cameraLabels = std::move(ocCameraLabels);
        onlineCalibration.imuLabels = std::move(ocImuLabels);
      } else {
        if (ocCameraSerialNumbers != onlineCalibration.cameraSerialNumbers ||
            ocCameraLabels != onlineCalibration.cameraLabels ||
            ocImuLabels != onlineCalibration.imuLabels) {
          throw std::runtime_error("mismatch in labels/serials");
        }
      }
    }

    std::cout << "Calib states: " << onlineCalibration.calibs.size() << "\n"
              << showTimeStamps(onlineCalibration.calibs) << std::endl;
  }

#if USE_OPEN_LOOP
  // load trajectory
  const auto ft = mps::readOpenLoopTrajectory(sessDataPath / kOpenLoopTrajectory);
  const Sophus::SE3d T_device_bodyImu = slamInfo.T_bodyImu_device.inverse();
  for (const auto& pose : ft) {
    inertialPoses.push_back(
        InertialPoseState{
            .T_w_IMU = pose.T_odometry_device * T_device_bodyImu,
            .v_w = pose.deviceLinearVelocity_odometry +
                pose.T_odometry_device.so3() *
                    pose.angularVelocity_device.cross(T_device_bodyImu.translation()),
            .omega_bodyImu = slamInfo.T_bodyImu_device.so3() * pose.angularVelocity_device,
            .timestamp_us = pose.trackingTimestamp.count(),
            .utc_timestamp_ns = pose.utcTimestamp.count(),
            .qualityScore = pose.qualityScore,
            .sessionOrGraphUid = pose.sessionUid,
        });
  }
#else
  // load trajectory
  const auto ft = mps::readClosedLoopTrajectory(sessDataPath / kClosedLoopTrajectory);
  const Sophus::SE3d T_device_bodyImu = slamInfo.T_bodyImu_device.inverse();
  for (const auto& pose : ft) {
    inertialPoses.push_back(
        InertialPoseState{
            .T_w_IMU = pose.T_world_device * T_device_bodyImu,
            .v_w = pose.T_world_device.so3() * pose.deviceLinearVelocity_device +
                pose.T_world_device.so3() *
                    pose.angularVelocity_device.cross(T_device_bodyImu.translation()),
            .omega_bodyImu = slamInfo.T_bodyImu_device.so3() * pose.angularVelocity_device,
            .timestamp_us = pose.trackingTimestamp.count(),
            .utc_timestamp_ns = pose.utcTimestamp.count(),
            .qualityScore = pose.qualityScore,
            .sessionOrGraphUid = pose.graphUid,
        });
  }
#endif
  std::cout << "Inertial poses: " << inertialPoses.size() << "\n"
            << showTimeStamps(inertialPoses) << std::endl;

  // load tracking observations
  trackingObservations = PointObservationReader::read(sessDataPath / kPointObservations);
  if (trackingObservations.empty()) {
    throw std::runtime_error("unable to load tracking observations");
  }
  std::cout << "Tracking Observations: " << trackingObservations.size() << "\n"
            << showTimeStamps3(trackingObservations) << std::endl;

  // load IMU measurements for all IMUs
  if (loadImuMeasurements) {
    for (const auto& imuLabel : slamInfo.imuLabels) {
      const auto imuPath = sessDataPath / fmt::format(kImuSamplesPattern, imuLabel);
      std::cout << "Loading IMU measurements for " << imuLabel << std::endl;
      allImuMeasurements.push_back(::imu_types::ImuDataReader::read(imuPath));
      std::cout << "n.meas: " << allImuMeasurements.back().size()
                << ", meas = " << showTimeStamps4(allImuMeasurements.back()) << std::endl;
    }
  }

  // load "reset_events.json", if it exists
  {
    const std::filesystem::path resetEventsPath = sessDataPath / "reset_events.json";
    if (std::filesystem::exists(resetEventsPath)) {
      std::ifstream ifile(resetEventsPath);
      if (ifile.is_open()) {
        auto json = nlohmann::json::parse(ifile);
        XR_CHECK(json.contains("reset_events"));
        XR_CHECK(json["reset_events"].is_array());
        const auto& jResetEvents = json["reset_events"];
        for (const auto& jRe : jResetEvents) {
          XR_CHECK(jRe.is_object());
          XR_CHECK(jRe.contains("tracking_timestamp_us"));
          XR_CHECK(jRe["tracking_timestamp_us"].is_number_integer());
          resetTimeStampsUs.push_back(jRe["tracking_timestamp_us"].get<int64_t>());
        }
      }
      if (verbose) {
        XR_LOGI("Reset events: {}", resetTimeStampsUs.size());
      }
    }
  }
}

} // namespace visual_inertial_ba
