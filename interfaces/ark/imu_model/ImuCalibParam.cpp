/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <imu_model/ImuCalibParam.h>

#define DEFAULT_LOG_CHANNEL "ViBa::ImuCalibParam"
#include <logging/Log.h>

namespace visual_inertial_ba {

ImuCalibrationOptions kDefaultImuCalibEstimationOptions{
    /*accelB = */ true,
    /*gyroB = */ true,
    /*accelSc = */ true,
    /*gyroSc = */ true,
    /*gyroNonO = */ true,
    /*accelNonO = */ true,
    /*refImuTimeOffset = */ true,
    /*gyroAccelTimeOffset = */ true,
};

void printEstOptionsVsDefaults(
    const std::string& message,
    const ImuCalibrationOptions& opt,
    const ImuCalibrationOptions& dft) {
  auto printWithDefault = [&](const std::string& label, bool val, bool defaultVal) -> std::string {
    std::stringstream ss;
    ss << label << ": ";
    if (val == defaultVal) {
      ss << (val ? "true" : "false") << " (=default)";
    } else {
      ss << (val ? "TRUE" : "FALSE") << " (default: " << (defaultVal ? "true" : "false") << ")";
    }
    return ss.str();
  };
#define P(field) printWithDefault(#field, opt.field, dft.field)
  XR_LOGI(
      "{}:\n  {}\n  {}\n  {}\n  {}\n  {}\n  {}\n  {}\n  {}",
      message,
      P(accelBias),
      P(gyroBias),
      P(accelScale),
      P(gyroScale),
      P(gyroNonOrth),
      P(accelNonOrth),
      P(referenceImuTimeOffset),
      P(gyroAccelTimeOffset));
#undef P
}

void ImuCalibParam::boxPlus(Eigen::Ref<const Eigen::VectorXd> correction) {
  XR_CHECK_EQ(correction.size(), jacInd->getErrorStateSize());

  if (estOpts->gyroBias) {
    modelParams.gyroBiasRadSec += correction.template segment<3>(jacInd->gyroBiasIdx());
  }

  if (estOpts->accelBias) {
    modelParams.accelBiasMSec2 += correction.template segment<3>(jacInd->accelBiasIdx());
  }

  if (estOpts->gyroScale) {
    Eigen::Vector3d invScale = modelParams.gyroScaleVec.cwiseInverse();
    invScale += correction.template segment<3>(jacInd->gyroScaleIdx());
    modelParams.gyroScaleVec = invScale.cwiseInverse();
  }

  if (estOpts->accelScale) {
    Eigen::Vector3d invScale = modelParams.accelScaleVec.cwiseInverse();
    invScale += correction.template segment<3>(jacInd->accelScaleIdx());
    modelParams.accelScaleVec = invScale.cwiseInverse();
  }

  if (estOpts->gyroNonOrth) {
    modelParams.gyroNonorth.row(0).template segment<2>(1) +=
        correction.template segment<2>(jacInd->gyroNonorthIdx());
    modelParams.gyroNonorth(1, 0) += correction(jacInd->gyroNonorthIdx() + 2);
    modelParams.gyroNonorth(1, 2) += correction(jacInd->gyroNonorthIdx() + 3);
    modelParams.gyroNonorth.row(2).template segment<2>(0) +=
        correction.template segment<2>(jacInd->gyroNonorthIdx() + 4);

    modelParams.gyroNonorth(0, 0) =
        std::sqrt(1.0 - modelParams.gyroNonorth.row(0).template segment<2>(1).squaredNorm());
    modelParams.gyroNonorth(1, 1) = std::sqrt(
        1.0 - modelParams.gyroNonorth(1, 0) * modelParams.gyroNonorth(1, 0) -
        modelParams.gyroNonorth(1, 2) * modelParams.gyroNonorth(1, 2));
    modelParams.gyroNonorth(2, 2) =
        std::sqrt(1.0 - modelParams.gyroNonorth.row(2).template segment<2>(0).squaredNorm());
  }

  if (estOpts->accelNonOrth) {
    modelParams.accelNonorth.row(0).template segment<2>(1) +=
        correction.template segment<2>(jacInd->accelNonorthIdx());
    modelParams.accelNonorth(1, 2) += correction(jacInd->accelNonorthIdx() + 2);

    modelParams.accelNonorth(0, 0) =
        std::sqrt(1.0 - modelParams.accelNonorth.row(0).template segment<2>(1).squaredNorm());

    modelParams.accelNonorth(1, 1) =
        std::sqrt(1.0 - modelParams.accelNonorth(1, 2) * modelParams.accelNonorth(1, 2));
    modelParams.accelNonorth(2, 2) = 1.0;
  }

  if (estOpts->referenceImuTimeOffset) {
    modelParams.dtReferenceGyroSec += correction(jacInd->referenceImuTimeOffsetIdx());
    modelParams.dtReferenceAccelSec += correction(jacInd->referenceImuTimeOffsetIdx());
  }

  if (estOpts->gyroAccelTimeOffset) {
    modelParams.dtReferenceAccelSec += correction(jacInd->gyroAccelTimeOffsetIdx());
  }
}

// compute the residual to another state (aka boxMinus)
void ImuCalibParam::boxMinus(
    const ImuMeasurementModelParameters& refModelParams,
    Eigen::Ref<Eigen::VectorXd> res) const {
  XR_CHECK_EQ(res.size(), jacInd->getErrorStateSize());

  if (estOpts->gyroBias) {
    res.template segment<3>(jacInd->gyroBiasIdx()) =
        modelParams.gyroBiasRadSec - refModelParams.gyroBiasRadSec;
  }

  if (estOpts->accelBias) {
    res.template segment<3>(jacInd->accelBiasIdx()) =
        modelParams.accelBiasMSec2 - refModelParams.accelBiasMSec2;
  }

  if (estOpts->gyroScale) {
    res.template segment<3>(jacInd->gyroScaleIdx()) =
        modelParams.gyroScaleVec.cwiseInverse() - refModelParams.gyroScaleVec.cwiseInverse();
  }

  if (estOpts->accelScale) {
    res.template segment<3>(jacInd->accelScaleIdx()) =
        modelParams.accelScaleVec.cwiseInverse() - refModelParams.accelScaleVec.cwiseInverse();
  }

  if (estOpts->gyroNonOrth) {
    res.template segment<2>(jacInd->gyroNonorthIdx()) =
        modelParams.gyroNonorth.row(0).template segment<2>(1) -
        refModelParams.gyroNonorth.row(0).template segment<2>(1);

    res(jacInd->gyroNonorthIdx() + 2) =
        modelParams.gyroNonorth(1, 0) - refModelParams.gyroNonorth(1, 0);
    res(jacInd->gyroNonorthIdx() + 3) =
        modelParams.gyroNonorth(1, 2) - refModelParams.gyroNonorth(1, 2);

    res.template segment<2>(jacInd->gyroNonorthIdx() + 4) =
        modelParams.gyroNonorth.row(2).template segment<2>(0) -
        refModelParams.gyroNonorth.row(2).template segment<2>(0);
  }

  if (estOpts->accelNonOrth) {
    res.template segment<2>(jacInd->accelNonorthIdx()) =
        modelParams.accelNonorth.row(0).template segment<2>(1) -
        refModelParams.accelNonorth.row(0).template segment<2>(1);

    res(jacInd->accelNonorthIdx() + 2) =
        modelParams.accelNonorth(1, 2) - refModelParams.accelNonorth(1, 2);
  }

  if (estOpts->referenceImuTimeOffset) {
    res(jacInd->referenceImuTimeOffsetIdx()) =
        modelParams.dtReferenceGyroSec - refModelParams.dtReferenceGyroSec;
  }

  if (estOpts->gyroAccelTimeOffset) {
    res(jacInd->gyroAccelTimeOffsetIdx()) =
        (modelParams.dtReferenceAccelSec - modelParams.dtReferenceGyroSec) -
        (refModelParams.dtReferenceAccelSec - refModelParams.dtReferenceGyroSec);
  }
}

void ImuCalibParam::eachNamedDeltaComponent(
    Eigen::Ref<const Eigen::VectorXd> delta,
    const std::function<void(const std::string&, Eigen::Ref<const Eigen::VectorXd>)>& enumFunc)
    const {
  if (estOpts->gyroBias) {
    enumFunc("1_GyroBiasRadSec", delta.segment<3>(jacInd->gyroBiasIdx()));
  }
  if (estOpts->accelBias) {
    enumFunc("2_AccelBiasMSec2", delta.segment<3>(jacInd->accelBiasIdx()));
  }
  if (estOpts->gyroScale) {
    enumFunc("3_GyroScale", delta.segment<3>(jacInd->gyroScaleIdx()));
  }
  if (estOpts->accelScale) {
    enumFunc("4_AccelScale", delta.segment<3>(jacInd->accelScaleIdx()));
  }
  if (estOpts->gyroNonOrth) {
    enumFunc("5_GyroNonOrth", delta.segment<6>(jacInd->gyroNonorthIdx()));
  }
  if (estOpts->accelNonOrth) {
    enumFunc("6_AccelNonOrth", delta.segment<3>(jacInd->accelNonorthIdx()));
  }
  if (estOpts->referenceImuTimeOffset) {
    enumFunc("7_RefImuTimeOffset", delta.segment<1>(jacInd->referenceImuTimeOffsetIdx()));
  }
  if (estOpts->gyroAccelTimeOffset) {
    enumFunc("8_GyroAccelTimeOffset", delta.segment<1>(jacInd->gyroAccelTimeOffsetIdx()));
  }
}

} // namespace visual_inertial_ba

namespace small_thing {

void VarSpec<visual_inertial_ba::ImuCalibParam>::getData(
    const DataType& value,
    Eigen::Ref<Eigen::VectorXd> data) {
  XR_CHECK_EQ(data.size(), visual_inertial_ba::kImuCalibDataDim);
  const auto& modelParams = value.modelParams;

  data.segment(0, 3) = modelParams.gyroScaleVec;
  data.segment(3, 3) = modelParams.accelScaleVec;
  data.segment(6, 3) = modelParams.gyroBiasRadSec;
  data.segment(9, 3) = modelParams.accelBiasMSec2;
  data.segment(12, 9) = Eigen::Map<const Eigen::Vector<double, 9>>(modelParams.gyroNonorth.data());
  data.segment(21, 9) = Eigen::Map<const Eigen::Vector<double, 9>>(modelParams.accelNonorth.data());
  data[30] = modelParams.dtReferenceAccelSec;
  data[31] = modelParams.dtReferenceGyroSec;
}

void VarSpec<visual_inertial_ba::ImuCalibParam>::setData(
    DataType& value,
    Eigen::Ref<const Eigen::VectorXd> data) {
  XR_CHECK_EQ(data.size(), visual_inertial_ba::kImuCalibDataDim);
  auto& modelParams = value.modelParams;

  modelParams.gyroScaleVec = data.segment(0, 3);
  modelParams.accelScaleVec = data.segment(3, 3);
  modelParams.gyroBiasRadSec = data.segment(6, 3);
  modelParams.accelBiasMSec2 = data.segment(9, 3);
  Eigen::Map<Eigen::Vector<double, 9>>(modelParams.gyroNonorth.data()) = data.segment(12, 9);
  Eigen::Map<Eigen::Vector<double, 9>>(modelParams.accelNonorth.data()) = data.segment(21, 9);
  modelParams.dtReferenceAccelSec = data[30];
  modelParams.dtReferenceGyroSec = data[31];
}

} // namespace small_thing
