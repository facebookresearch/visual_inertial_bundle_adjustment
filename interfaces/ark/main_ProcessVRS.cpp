/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <imu_types/ImuDataWriter.h>
#include <projectaria_tools/core/calibration/loader/DeviceCalibrationJson.h>
#include <projectaria_tools/core/data_provider/VrsDataProvider.h>

#include <CLI/CLI.hpp>

using namespace imu_types;
using namespace projectaria::tools;
using vrs::StreamId;

constexpr const char* kFactoryCalibration = "factory_calibration.json";
constexpr const char* kImuSamplesPattern = "imu_samples_{}.csv";

int main(int argc, const char* argv[]) {
  std::filesystem::path vrsPathIn;
  std::filesystem::path folderOut;

  CLI::App app{"Process VRS extracting IMU + FactoryCalibration"};
  app.add_option("-i,--in", vrsPathIn, "VRS input")->required();
  app.add_option("-o,--out", folderOut, "Output directory path (will be created)")->required();

  CLI11_PARSE(app, argc, argv);

  // Open the VRS File
  auto dataProvider = data_provider::createVrsDataProvider(vrsPathIn);
  if (!dataProvider) {
    std::cout << "Error, unable to open: " << vrsPathIn << std::endl;
    return 0;
  }

  std::filesystem::create_directories(folderOut);

  auto allStreams = dataProvider->getAllStreams();
  std::map<StreamId, std::string> streamToLabel;
  std::map<StreamId, std::unique_ptr<ImuDataWriter>> streamToImuWriter;
  for (const auto& sId : allStreams) {
    auto maybeLabel = dataProvider->getLabelFromStreamId(sId);
    std::string label = "<none>";
    if (maybeLabel.has_value()) {
      label = *maybeLabel;
    }
    std::cout << "Stream " << sId.getFullName() << ": " << label //
              << ", type = " << (int)sId.getTypeId() << std::endl;
    streamToLabel[sId] = label;

    auto dataType = dataProvider->getSensorDataType(sId);
    if (dataType == data_provider::SensorDataType::Imu) {
      streamToImuWriter[sId] =
          std::make_unique<ImuDataWriter>(folderOut / fmt::format(kImuSamplesPattern, label));
    }
  }

  auto maybeDeviceCalib = dataProvider->getDeviceCalibration();
  if (maybeDeviceCalib.has_value()) {
    std::cout << "Got device calib!" << std::endl;
    std::ofstream of(folderOut / kFactoryCalibration);
    of << calibration::deviceCalibrationToJson(*maybeDeviceCalib);
  } else {
    std::cout << "No device calib..." << std::endl;
  }

  int64_t totData = 0, imuData = 0, noLabel = 0;
  for (const auto& data : dataProvider->deliverQueuedSensorData()) {
    totData++;
    if (data.sensorDataType() == data_provider::SensorDataType::Imu) {
      imuData++;
      auto it = streamToImuWriter.find(data.streamId());
      if (it == streamToImuWriter.end()) {
        noLabel++;
        continue;
      }
      const data_provider::MotionData& m = data.imuData();
      it->second->write(
          ImuMeasurement{
              .timestampNs = m.captureTimestampNs,
              .temperatureC = m.temperature,
              .accelMSec2 = Eigen::Vector3f(m.accelMSec2.data()).cast<double>(),
              .gyroRadSec = Eigen::Vector3f(m.gyroRadSec.data()).cast<double>(),
          });
    }
  }
  std::cout << "tot data: " << totData << ", imu: " << imuData << ", noLab: " << noLabel
            << std::endl;
}
