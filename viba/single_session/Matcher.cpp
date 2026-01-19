/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/single_session/Matcher.h>

#include <viba/common/Enumerate.h>
#include <iostream>

#include <logging/Checks.h>
#define DEFAULT_LOG_CHANNEL "ViBa::Matcher"
#include <logging/Log.h>

namespace visual_inertial_ba {

void Matcher::buildIndices(SessionData& fData, bool verbose) {
  // build map time stamp -> index in evolving states
  std::unordered_map<int64_t, int64_t> iPosesTsToIndex;
  for (const auto& [i, es] : enumerate(fData.inertialPoses)) {
    iPosesTsToIndex[es.timestamp_us] = i;
  }

  // build map time stamp -> index in calib states
  std::unordered_map<int64_t, int64_t> calibStatesTsToIndex;
  for (const auto& [i, cs] : enumerate(fData.onlineCalibration.calibs)) {
    calibStatesTsToIndex[cs.timestamp_us] = i;
  }

  // sort referenced time stamps, rig indices will be the index
  std::vector<int64_t> sortedTs;
  for (const auto& [ts, i] : calibStatesTsToIndex) {
    if (iPosesTsToIndex.count(ts)) {
      sortedTs.push_back(ts);
    }
  }
  std::sort(sortedTs.begin(), sortedTs.end());
  if (verbose) {
    XR_LOGI("Rig indices... tot rigs: {}", sortedTs.size());
    XR_LOGI(
        "Discarded {} only in calibs, and {} only in trajectory",
        calibStatesTsToIndex.size() - sortedTs.size(),
        iPosesTsToIndex.size() - sortedTs.size());
  }

  // build ts -> index map, and indices to ev states and calib states
  rigIndexToEvolvingStateIndex.reserve(sortedTs.size());
  rigIndexToCalibStateIndex.reserve(sortedTs.size());

  if (verbose) {
    XR_LOGI("Rig indices...");
  }
  for (const auto& [i, ts] : enumerate(sortedTs)) {
    timestampToRigIndex[ts] = i;
    rigIndexToEvolvingStateIndex.push_back(iPosesTsToIndex.at(ts));
    rigIndexToCalibStateIndex.push_back(calibStatesTsToIndex.at(ts));
  }

  if (verbose) {
    XR_LOGI("Build obs...");
  }
  // indices from obs to referenced rig
  int skippedObsReferences = 0;
  processedObsToRigIndex.assign(fData.trackingObservations.size(), -1);
  for (const auto& [i, po] : enumerate(fData.trackingObservations)) {
    auto rigIndexIt = timestampToRigIndex.find(po.captureTimestampUs);
    if (rigIndexIt != timestampToRigIndex.end()) {
      processedObsToRigIndex[i] = rigIndexIt->second;
    } else {
      skippedObsReferences++;
    }
  }

  // Warn for missing rig in evolving/calib states
  if (verbose) {
    XR_LOGI("Losing references from {} observations", skippedObsReferences);
  }

  // build point indices
  for (const auto& [i, po] : enumerate(fData.trackingObservations)) {
    if (processedObsToRigIndex[i] < 0) {
      continue;
    }
    auto it = pointIdToPointIndex.find(po.pointId);
    if (it == pointIdToPointIndex.end()) {
      int64_t newIndex = pointIdToPointIndex.size();
      pointIdToPointIndex[po.pointId] = newIndex;
    }
  }

  // for each point index, populate tracks
  pointIndexToObsIndices.resize(pointIdToPointIndex.size());
  for (const auto& [i, po] : enumerate(fData.trackingObservations)) {
    if (processedObsToRigIndex[i] < 0) {
      continue;
    }
    pointIndexToObsIndices[pointIdToPointIndex[po.pointId]].push_back(i);
  }

  // create set of reset rig indices
  for (int64_t resetTs : fData.resetTimeStampsUs) {
    auto it = timestampToRigIndex.find(resetTs);
    if (it != timestampToRigIndex.end()) {
      resetRigIndices.insert(it->second);
    } else { // reset not found? then add index of last rig BEFORE that timestamp
      int64_t foundRigTs = -1;
      int64_t foundRigIndex = -1;
      for (auto [rigIndex, calibIndex] : enumerate(rigIndexToCalibStateIndex)) {
        const int64_t rigTs = fData.onlineCalibration.calibs[calibIndex].timestamp_us;
        if (rigTs > foundRigTs && rigTs < resetTs) {
          foundRigTs = rigTs;
          foundRigIndex = rigIndex;
        }
      }
      if (foundRigIndex >= 0) {
        resetRigIndices.insert(foundRigIndex);
      }
    }
  }

  // match camera/imu labels
  slamCamIndexToFactoryCalib.reserve(fData.slamInfo.cameraSerialNumbers.size());
  slamCamIndexToOnlineCalib.reserve(fData.slamInfo.cameraSerialNumbers.size());
  for (const auto& serialNumber : fData.slamInfo.cameraSerialNumbers) {
    auto it = std::find(
        fData.factoryCalibration.cameraSerialNumbers.begin(),
        fData.factoryCalibration.cameraSerialNumbers.end(),
        serialNumber);
    XR_CHECK(
        it != fData.factoryCalibration.cameraSerialNumbers.end(),
        "Camera serial number not found in factory calibration: {}",
        serialNumber);
    slamCamIndexToFactoryCalib.push_back(
        std::distance(fData.factoryCalibration.cameraSerialNumbers.begin(), it));

    auto it2 = std::find(
        fData.onlineCalibration.cameraSerialNumbers.begin(),
        fData.onlineCalibration.cameraSerialNumbers.end(),
        serialNumber);
    XR_CHECK(
        it2 != fData.onlineCalibration.cameraSerialNumbers.end(),
        "Camera serial number not found in online calibration: {}",
        serialNumber);
    slamCamIndexToOnlineCalib.push_back(
        std::distance(fData.onlineCalibration.cameraSerialNumbers.begin(), it2));
  }

  slamImuIndexToFactoryCalib.reserve(fData.slamInfo.imuLabels.size());
  slamImuIndexToOnlineCalib.reserve(fData.slamInfo.imuLabels.size());
  for (const auto& label : fData.slamInfo.imuLabels) {
    auto it = std::find(
        fData.factoryCalibration.imuLabels.begin(),
        fData.factoryCalibration.imuLabels.end(),
        label);
    XR_CHECK(
        it != fData.factoryCalibration.imuLabels.end(),
        "Imu label number not found in factory calibration: {}",
        label);
    slamImuIndexToFactoryCalib.push_back(
        std::distance(fData.factoryCalibration.imuLabels.begin(), it));

    auto it2 = std::find(
        fData.onlineCalibration.imuLabels.begin(), fData.onlineCalibration.imuLabels.end(), label);
    XR_CHECK(
        it2 != fData.onlineCalibration.cameraSerialNumbers.end(),
        "Imu label number not found in online calibration: {}",
        label);
    slamImuIndexToOnlineCalib.push_back(
        std::distance(fData.onlineCalibration.imuLabels.begin(), it2));
  }

  if (verbose) {
    XR_LOGI("Matching done!");
  }
}

} // namespace visual_inertial_ba
