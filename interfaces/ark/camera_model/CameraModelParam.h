/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <projectaria_tools/core/calibration/CameraCalibration.h>
#include <small_thing/Variable.h>

namespace visual_inertial_ba {
// most complex model (Fisheye624) has 15 parameters:
//   1 focal length + 2 (principal points) + 6 (K) + 2 (tangential) + 4 (thin prism)
// We have to add 2: readout time and time offset
constexpr int kMaxCamParams = 17;

using namespace projectaria::tools::calibration;

struct CameraModelParam {
  // for internal use, used in SessionData.cpp
  CameraModelParam(
      const CameraCalibration& modelParams,
      bool estimateReadoutTime,
      bool estimateTimeOffset);

  // interface, copy constructor
  CameraModelParam(const CameraModelParam& other)
      : model(other.model),
        estimateReadoutTime(other.estimateReadoutTime),
        estimateTimeOffset(other.estimateTimeOffset) {}

  // interface method
  bool project(
      const Eigen::Vector3d& pointInCamera,
      Eigen::Ref<Eigen::Vector2d> pointPixel,
      Eigen::Ref<Eigen::Matrix<double, 2, 3>> J_proj_cam = NullRef(),
      Eigen::Ref<Eigen::Matrix<double, 2, Eigen::Dynamic>> J_proj_params = NullRef()) const {
    constexpr bool kUseAriaKitChecks = false;
    if (kUseAriaKitChecks) {
      auto maybePointPixel = model.project(pointInCamera, J_proj_cam, J_proj_params);
      if (maybePointPixel.has_value()) {
        pointPixel = *maybePointPixel;
        return true;
      }
      return false;
    } else {
      if (pointInCamera[2] < 1e-6) {
        return false;
      }
      pointPixel = model.projectNoChecks(pointInCamera, J_proj_cam, J_proj_params);
      return true;
    }
  }

  // interface method
  Eigen::Vector3d unprojectNoChecks(const Eigen::Vector2d& pointPixel) const {
    return model.unprojectNoChecks(pointPixel);
  }

  // interface method
  Eigen::VectorXd intrinsicParams() const {
    return model.projectionParams();
  }

  // interface method
  int numModelParameters() const {
    return model.numParameters();
  }

  // interface method
  int imageWidth() const {
    return model.getImageSize().x();
  }

  // interface method
  int imageHeight() const {
    return model.getImageSize().y();
  }

  // interface method
  double timeOffsetSec_Dev_Camera() const {
    return model.getTimeOffsetSecDeviceCamera();
  }

  // interface method
  double readoutTimeSec() const {
    return model.getReadOutTimeSec().value_or(0.0);
  }

  // interface method
  bool hasTimeOffset() const {
    return estimateTimeOffset || model.getTimeOffsetSecDeviceCamera() != 0.0;
  }

  // interface method
  bool isRollingShutter() const {
    return estimateReadoutTime || model.getReadOutTimeSec().has_value();
  }

  // interface method
  void setEstimateReadoutTime(bool value) {
    estimateReadoutTime = value;
  }

  // interface method
  void setEstimateTimeOffset(bool value) {
    estimateTimeOffset = value;
  }

  // interface method, just for evaluation of calibration
  void eachNamedDeltaComponent(
      Eigen::Ref<const Eigen::VectorXd> delta,
      const std::function<void(const std::string&, Eigen::Ref<const Eigen::VectorXd>)>&) const;

  CameraCalibration model;
  bool estimateReadoutTime = false;
  bool estimateTimeOffset = false;
};

} // namespace visual_inertial_ba

namespace small_thing {
template <>
struct VarSpec<visual_inertial_ba::CameraModelParam> {
  using DataType = visual_inertial_ba::CameraModelParam;
  static constexpr int DataDimSpec = Eigen::Dynamic;
  static constexpr int TangentDimSpec = Eigen::Dynamic;

  static int getDynamicDataDim(const DataType& value) {
    return value.model.numParameters() + //
        (value.estimateReadoutTime ? 1 : 0) + //
        (value.estimateTimeOffset ? 1 : 0);
  }

  static int getDynamicTangentDim(const DataType& value) {
    return getDynamicDataDim(value);
  }

  static double applyBoxPlus(DataType& value, Eigen::Ref<const Eigen::VectorXd> step);

  static void
  boxMinus(const DataType& value, const DataType& base, Eigen::Ref<Eigen::VectorXd> delta);

  static void getData(const DataType& value, Eigen::Ref<Eigen::VectorXd> data);

  static void setData(DataType& value, Eigen::Ref<const Eigen::VectorXd> data);
};
} // namespace small_thing
