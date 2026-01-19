/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <preintegration/ImuUtils.h>

#include <fmt/format.h>

namespace visual_inertial_ba::preintegration {

void normalizeImuParams(ImuMeasurementModelParameters& modelParams) {
  modelParams.gyroNonorth(0, 0) =
      std::sqrt(1.0 - modelParams.gyroNonorth.row(0).template segment<2>(1).squaredNorm());
  modelParams.gyroNonorth(1, 1) = std::sqrt(
      1.0 - modelParams.gyroNonorth(1, 0) * modelParams.gyroNonorth(1, 0) -
      modelParams.gyroNonorth(1, 2) * modelParams.gyroNonorth(1, 2));
  modelParams.gyroNonorth(2, 2) =
      std::sqrt(1.0 - modelParams.gyroNonorth.row(2).template segment<2>(0).squaredNorm());

  modelParams.accelNonorth(0, 0) =
      std::sqrt(1.0 - modelParams.accelNonorth.row(0).template segment<2>(1).squaredNorm());
  modelParams.accelNonorth(1, 1) =
      std::sqrt(1.0 - modelParams.accelNonorth(1, 2) * modelParams.accelNonorth(1, 2));
  modelParams.accelNonorth(2, 2) = 1.0;
  modelParams.accelNonorth(1, 0) = 0;
  modelParams.accelNonorth(2, 0) = 0;
  modelParams.accelNonorth(2, 1) = 0;
}

ImuMeasurementModelParameters factoryImuParams() {
  ImuMeasurementModelParameters p;

  p.gyroScaleVec << 0.9975922107696533, 0.9992708563804626, 1.002429008483887;
  p.accelScaleVec << 1.00313138961792, 0.9989509582519531, 1.00210428237915; //
  p.accelBiasMSec2 << -0.03137952834367752, -0.1199406236410141, 0.04399538785219193; //
  p.gyroBiasRadSec << 0.001339819049462676, 0.0001755904668243602, -0.001454736455343664; //
  p.accelNonorth << 1, 1.941762820933945e-05, -0.000217802997212857, //
      0, 0.9999973177909851, -0.002304504392668605, //
      0, 0, 1;
  p.gyroNonorth << 0.9999923706054688, -0.003904011566191912, 4.595328937284648e-05, //
      0.003785371780395508, 0.999991774559021, -0.001455615041777492, //
      0.0010240338742733, 0.003811037633568048, 0.9999921917915344;
  p.dtReferenceAccelSec = 0.002755688270553946;
  p.dtReferenceGyroSec = 0.004112016409635544;

  normalizeImuParams(p);
  return p;
}

std::string printableImuCalibJacDelta(
    Ref<const MatX> J,
    const ImuCalibrationJacobianIndices& jacInd,
    const std::vector<std::tuple<std::string, int, int>>& arrivalRanges,
    bool printBlocks,
    double epsilonCheck) {
  if (J.cols() != jacInd.getErrorStateSize()) {
    throw std::runtime_error("printableImuCalibJacDelta: incorrect Jacobian size");
  }

  static const char* OK = "\033[32mOK\033[0m";
  static const char* FAIL = "\033[31mFAIL\033[0m";
  std::stringstream ss;

  for (const auto& range : arrivalRanges) {
    auto show = [&](const char* field, int start, int size) {
      if (start < 0) {
        return;
      }
      auto block = J.block(std::get<1>(range), start, std::get<2>(range), size);
      const double normInf = block.cwiseAbs().maxCoeff();
      ss << fmt::format(
          " [{}] |d({})/d({})| ͚ = {}\n",
          normInf < epsilonCheck ? OK : FAIL,
          std::get<0>(range),
          field,
          normInf);
      if (printBlocks) {
        ss << block.transpose() << std::endl;
      }
    };

#define ShowField(FIELD, DOFS) show(#FIELD, jacInd.FIELD##Idx(), DOFS)

    ShowField(gyroBias, 3);
    ShowField(accelBias, 3);
    ShowField(gyroScale, 3);
    ShowField(accelScale, 3);
    ShowField(gyroNonorth, 6);
    ShowField(accelNonorth, 3);
    ShowField(referenceImuTimeOffset, 1);
    ShowField(gyroAccelTimeOffset, 1);

#undef ShowField
  }

  return ss.str();
}

std::string printableImuMeasJacDelta(
    Ref<const MatX6> J,
    const std::vector<std::tuple<std::string, int, int>>& arrivalRanges,
    bool printBlocks,
    double epsilonCheck) {
  static const char* OK = "\033[32mOK\033[0m";
  static const char* FAIL = "\033[31mFAIL\033[0m";
  std::stringstream ss;

  for (const auto& range : arrivalRanges) {
    auto show = [&](const char* field, int start, int size) {
      if (start < 0) {
        return;
      }
      auto block = J.block(std::get<1>(range), start, std::get<2>(range), size);
      const double normInf = block.cwiseAbs().maxCoeff();
      ss << fmt::format(
          " [{}] |d({})/d({})| ͚ = {}\n",
          normInf < epsilonCheck ? OK : FAIL,
          std::get<0>(range),
          field,
          normInf);
      if (printBlocks) {
        ss << block.transpose() << std::endl;
      }
    };

    show("rawGyro", 0, 3);
    show("rawAccel", 3, 3);
  }

  return ss.str();
}

} // namespace visual_inertial_ba::preintegration
