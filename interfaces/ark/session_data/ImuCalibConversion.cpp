/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <session_data/ImuCalibConversion.h>
#include <iostream>

namespace visual_inertial_ba {

ImuMeasurementModelParameters fromProjectAriaCalibration(const ImuCalibration& calib) {
  ImuMeasurementModelParameters retv;
  retv.accelBiasMSec2 = calib.getAccelModel().getBias();
  retv.gyroBiasRadSec = calib.getGyroModel().getBias();
  retv.setScaleMatrices(
      calib.getGyroModel().getRectification(), calib.getAccelModel().getRectification());
  retv.dtReferenceAccelSec = calib.getTimeOffsetSecDeviceAccel();
  retv.dtReferenceGyroSec = calib.getTimeOffsetSecDeviceGyro();
  return retv;
}

ImuCalibration toProjectAriaCalibration(
    const ImuMeasurementModelParameters& calib,
    const std::string& label,
    const Sophus::SE3d& T_Device_Imu) {
  return ImuCalibration(
      label,
      calib.getAccelScaleMat(),
      calib.accelBiasMSec2,
      calib.getGyroScaleMat(),
      calib.gyroBiasRadSec,
      T_Device_Imu,
      calib.dtReferenceAccelSec,
      calib.dtReferenceGyroSec);
}

} // namespace visual_inertial_ba
