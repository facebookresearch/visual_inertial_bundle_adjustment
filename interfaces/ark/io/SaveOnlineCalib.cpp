/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <io/SaveOnlineCalib.h>

#include <calibration/loader/SensorCalibrationJson.h>
#include <nlohmann/json.hpp>
#include <session_data/ImuCalibConversion.h>
#include <viba/common/Enumerate.h>
#include <fstream>

#define DEFAULT_LOG_CHANNEL "ViBa::SaveOnlineCalib"
#include <logging/Log.h>

namespace visual_inertial_ba {

using namespace projectaria::tools::calibration;

void saveOnlineCalib(
    const SingleSessionProblem& prob,
    const SessionData& fData,
    const std::filesystem::path& outputPath) {
  const int numCameras = fData.slamInfo.cameraSerialNumbers.size();
  const int numImus = fData.slamInfo.imuLabels.size();
  auto rigIndices = prob.sortedRigIndices();

  std::ofstream ofs(outputPath);
  for (int64_t rigIndex : rigIndices) {
    nlohmann::json json;
    const auto& inputRig = fData.inertialPoses.at(rigIndex);
    json["tracking_timestamp_us"] = inputRig.timestamp_us;
    json["utc_timestamp_ns"] = inputRig.utc_timestamp_ns;

    nlohmann::json jCameras = nlohmann::json::array();
    for (int s = 0; s < numCameras; s++) { // set camera intrinsics
      CameraCalibration model = prob.cameraModel(rigIndex, s).var.value.model;
      SE3 T_Cam_BodyImu = prob.T_Cam_BodyImu(rigIndex, s).var.value;
      model.getT_Device_CameraMut() =
          fData.slamInfo.T_bodyImu_device.inverse() * T_Cam_BodyImu.inverse();
      jCameras.push_back(cameraCalibrationToJson(model));
    }
    json["CameraCalibrations"] = jCameras;

    nlohmann::json jImus = nlohmann::json::array();
    for (int s = 0; s < numImus; s++) {
      const auto& imuModel = prob.imuCalib(rigIndex, s).var.value.modelParams;
      SE3 T_Imu_BodyImu = s == 0 ? SE3() : prob.T_Imu_BodyImu(rigIndex, s).var.value;
      ImuCalibration imuCalib = toProjectAriaCalibration(
          imuModel,
          fData.slamInfo.imuLabels[s],
          fData.slamInfo.T_bodyImu_device.inverse() * T_Imu_BodyImu.inverse());
      jImus.push_back(imuCalibrationToJson(imuCalib));
    }
    json["ImuCalibrations"] = jImus;

    ofs << json.dump() << std::endl;
  }

  XR_LOGI("Online calibration ({} entries) saved to {}", rigIndices.size(), outputPath.string());
}

} // namespace visual_inertial_ba
