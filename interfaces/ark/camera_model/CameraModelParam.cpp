/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <camera_model/CameraModelParam.h>

#include <projectaria_tools/core/calibration/camera_projections/CameraProjectionFormat.h>

#define DEFAULT_LOG_CHANNEL "ViBa::CameraModelParam"
#include <logging/Checks.h>
#include <logging/Log.h>

namespace visual_inertial_ba {

CameraModelParam::CameraModelParam(
    const CameraCalibration& modelParams,
    bool estimateReadoutTime,
    bool estimateTimeOffset)
    : model(modelParams),
      estimateReadoutTime(estimateReadoutTime),
      estimateTimeOffset(estimateTimeOffset) {}

void CameraModelParam::eachNamedDeltaComponent(
    Eigen::Ref<const Eigen::VectorXd> delta,
    const std::function<void(const std::string&, Eigen::Ref<const Eigen::VectorXd>)>& enumFunc)
    const {
  XR_CHECK_EQ(delta.size(), ::small_thing::VarSpec<CameraModelParam>::getDynamicDataDim(*this));

  XR_CHECK(
      model.modelName() == CameraProjection::ModelType::Fisheye624,
      "Calib field enumeration unsupported for camera type {}",
      model.modelName());

  enumFunc("1_FocalLength", delta.segment(0, 1));
  enumFunc("2_PrincipalPt", delta.segment(1, 2));
  enumFunc("3_Distorsion", delta.segment(3, 6));
  enumFunc("4_Tangential", delta.segment(9, 2));
  enumFunc("5_ThinPrism", delta.segment(11, 4));
  if (estimateReadoutTime) {
    enumFunc("6_ReadoutTime", delta.segment(15, 1));
  }
  if (estimateTimeOffset) {
    enumFunc("7_TimeOffset", delta.tail(1));
  }
}

} // namespace visual_inertial_ba

namespace small_thing {

double VarSpec<visual_inertial_ba::CameraModelParam>::applyBoxPlus(
    DataType& value,
    Eigen::Ref<const Eigen::VectorXd> step) {
  int nIntr = value.model.numParameters();
  value.model.projectionParamsMut() += step.head(nIntr);
  if (value.estimateReadoutTime) {
    const double timeOffsetSec = value.model.getReadOutTimeSec().value_or(0.0);
    value.model.getReadOutTimeSecMut() = timeOffsetSec + step[nIntr++];
  }
  if (value.estimateTimeOffset) {
    value.model.getTimeOffsetSecDeviceCameraMut() += step[nIntr++];
  }
  return step.template lpNorm<Eigen::Infinity>();
}

void VarSpec<visual_inertial_ba::CameraModelParam>::boxMinus(
    const DataType& value,
    const DataType& base,
    Eigen::Ref<Eigen::VectorXd> delta) {
  int nIntr = value.model.numParameters();
  const int nData =
      nIntr + (value.estimateReadoutTime ? 1 : 0) + (value.estimateTimeOffset ? 1 : 0);
  XR_CHECK_EQ(delta.size(), nData);
  delta.head(nIntr) = value.model.projectionParams() - base.model.projectionParams();
  if (value.estimateReadoutTime) {
    delta[nIntr++] =
        value.model.getReadOutTimeSec().value() - base.model.getReadOutTimeSec().value();
  }
  if (value.estimateTimeOffset) {
    delta[nIntr++] =
        value.model.getTimeOffsetSecDeviceCamera() - base.model.getTimeOffsetSecDeviceCamera();
  }
}

void VarSpec<visual_inertial_ba::CameraModelParam>::getData(
    const DataType& value,
    Eigen::Ref<Eigen::VectorXd> data) {
  int nIntr = value.model.numParameters();
  const int nData =
      nIntr + (value.estimateReadoutTime ? 1 : 0) + (value.estimateTimeOffset ? 1 : 0);
  XR_CHECK_EQ(data.size(), nData);
  data.head(nIntr) = value.model.projectionParams();
  if (value.estimateReadoutTime) {
    data[nIntr++] = value.model.getReadOutTimeSec().value();
  }
  if (value.estimateTimeOffset) {
    data[nIntr++] = value.model.getTimeOffsetSecDeviceCamera();
  }
}

void VarSpec<visual_inertial_ba::CameraModelParam>::setData(
    DataType& value,
    Eigen::Ref<const Eigen::VectorXd> data) {
  int nIntr = value.model.numParameters();
  value.model.projectionParamsMut() = data.head(nIntr);
  if (value.estimateReadoutTime) {
    value.model.getReadOutTimeSecMut().value() = data[nIntr++];
  }
  if (value.estimateTimeOffset) {
    value.model.getTimeOffsetSecDeviceCameraMut() = data[nIntr++];
  }
}

} // namespace small_thing
