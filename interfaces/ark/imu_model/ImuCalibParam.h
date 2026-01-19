/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <imu_types/ImuCalibrationJacobianIndices.h>
#include <imu_types/ImuCalibrationOptions.h>
#include <imu_types/ImuMeasurementModelParameters.h>
#include <logging/Checks.h>
#include <small_thing/Variable.h>

namespace visual_inertial_ba {

using ImuCalibrationOptions = ::imu_types::ImuCalibrationOptions;
using ImuCalibrationJacobianIndices = ::imu_types::ImuCalibrationJacobianIndices;
using ImuMeasurementModelParameters = ::imu_types::ImuMeasurementModelParameters;

// Hardcoded max sizes.
constexpr int kImuCalibDataDim = 32;
constexpr int kImuCalibTangentMaxDim = ImuCalibrationJacobianIndices::kMaxCalibrationStateSize;

extern ImuCalibrationOptions kDefaultImuCalibEstimationOptions;

// interface
void printEstOptionsVsDefaults(
    const std::string& message,
    const ImuCalibrationOptions& opt,
    const ImuCalibrationOptions& dft);

struct ImuCalibParam {
  ImuMeasurementModelParameters modelParams;

  const ImuCalibrationOptions* estOpts = nullptr;
  const ImuCalibrationJacobianIndices* jacInd = nullptr;

  void boxPlus(Eigen::Ref<const Eigen::VectorXd> correction);

  // compute the residual to another state (aka boxMinus)
  void boxMinus(
      const ImuMeasurementModelParameters& refModelParams,
      Eigen::Ref<Eigen::VectorXd> res) const;

  // interface method, just for evaluation of calibration
  void eachNamedDeltaComponent(
      Eigen::Ref<const Eigen::VectorXd> delta,
      const std::function<void(const std::string&, Eigen::Ref<const Eigen::VectorXd>)>&) const;
};

} // namespace visual_inertial_ba

namespace small_thing {
template <>
struct VarSpec<visual_inertial_ba::ImuCalibParam> {
  static constexpr int DataDimSpec = visual_inertial_ba::kImuCalibDataDim;
  static constexpr int TangentDimSpec = Eigen::Dynamic;
  using DataType = visual_inertial_ba::ImuCalibParam;

  static int getDynamicTangentDim(const DataType& value) {
    return value.estOpts->errorStateSize();
  }

  static double applyBoxPlus(DataType& value, Eigen::Ref<const Eigen::VectorXd> step) {
    XR_CHECK_EQ(value.estOpts->errorStateSize(), step.size());
    value.boxPlus(step);
    return step.template lpNorm<Eigen::Infinity>();
  }

  static void
  boxMinus(const DataType& value, const DataType& base, Eigen::Ref<Eigen::VectorXd> delta) {
    value.boxMinus(base.modelParams, delta);
  }

  static void getData(const DataType& value, Eigen::Ref<Eigen::VectorXd> data);

  static void setData(DataType& value, Eigen::Ref<const Eigen::VectorXd> data);
};
} // namespace small_thing
