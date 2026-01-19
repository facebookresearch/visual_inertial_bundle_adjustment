/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/single_session/SingleSessionAdapter.h>

#define DEFAULT_LOG_CHANNEL "ViBa::EvalCalibration"
#include <logging/Log.h>

namespace visual_inertial_ba {

void SingleSessionAdapter::showCalibrationCompResults(
    const std::map<std::string, StatsValueContainer>& allStats) {
  std::stringstream ss;
  for (const auto& [key, cont] : allStats) {
    ss << key << ":\n  p50: " << cont.p50() << ", p90: " << cont.p90() << ", mean: " << cont.mean()
       << ", rmse: " << cont.rmse() << "\n";
  }
  std::cout << fmt::format("Calibration Eval:\n{}", ss.str()) << std::endl;
}

// NOTE: we only support eval vs factory calibration. We could add comparison with a time-varying GT
void SingleSessionAdapter::compareCalibrationVsFactory(
    std::map<std::string, StatsValueContainer>& allStats) const {
  // camera extrinsics
  for (auto [rigIndexCamIndex, varIndex] : prob_.rigCamToExtrIndex()) {
    auto [rigIndex, camIndex] = rigIndexCamIndex;
    SE3 errAtCam = prob_.T_Cam_BodyImu_at(varIndex).var.value *
        fData_.factoryCalibration.calib.T_Cam_BodyImu[matcher_.slamCamIndexToFactoryCalib[camIndex]]
            .inverse();
    std::stringstream label;
    label << "Cam" << camIndex << ".Extr.";
    allStats[label.str() + "RotDegs"].add(radiansToDegrees(errAtCam.so3().log().norm()));
    allStats[label.str() + "TrMMs"].add(errAtCam.translation().norm() * 1000.0);
  }

  // camera intrinsics
  std::unordered_set<int> camerasWithIncompatibleCalib;
  for (auto [rigIndexCamIndex, cameraModelIndex] : prob_.rigCamToModelIndex()) {
    auto [rigIndex, camIndex] = rigIndexCamIndex;
    auto gtModel = fData_.factoryCalibration.calib.getConvertedCameraModelParam(
        matcher_.slamCamIndexToFactoryCalib[camIndex]);
    const auto& varModel = prob_.cameraModel_at(cameraModelIndex).var.value;
    Eigen::VectorXd delta(::small_thing::VarSpec<CameraModelParam>::getDynamicTangentDim(varModel));
    ::small_thing::VarSpec<CameraModelParam>::boxMinus(varModel, gtModel, delta);

    std::string label = fmt::format("Cam{}.Intr.", camIndex);
    varModel.eachNamedDeltaComponent(
        delta, [&](const std::string& name, Eigen::Ref<const Eigen::VectorXd> segment) {
          allStats[label + name].add(segment.norm());
        });
  }

  // image projection offset (estimate vs GT), for points sampled at different distances
  int64_t nDone = 0;
  const std::vector<double> distances{0.5, 1.0, 2.0, 4.0, 8.0, 16.0};
  for (auto [rigIndexCamIndex, cameraModelIndex] : prob_.rigCamToModelIndex()) {
    auto [rigIndex, camIndex] = rigIndexCamIndex;
    int64_t extrIndex = findOrDie(prob_.rigCamToExtrIndex(), rigIndexCamIndex);
    auto gtModel = fData_.factoryCalibration.calib.getConvertedCameraModelParam(
        matcher_.slamCamIndexToFactoryCalib[camIndex]);
    const auto& varCamModel = prob_.cameraModel_at(cameraModelIndex).var.value;
    int16_t w = varCamModel.imageWidth();
    int16_t h = varCamModel.imageHeight();

    const auto& varT_Cam_BodyImu = prob_.T_Cam_BodyImu_at(extrIndex).var.value;
    const auto& gtT_Cam_BodyImu = fData_.factoryCalibration.calib
                                      .T_Cam_BodyImu[matcher_.slamCamIndexToFactoryCalib[camIndex]];

    static constexpr int stepX = 20, stepY = 15;
    double startX = fmod(w * 0.5, stepX);
    double startY = fmod(h * 0.5, stepY);
    for (double x = startX; x < w; x += stepX) {
      for (double y = startY; y < h; y += stepY) {
        Vec2 px{x, y};

        // check that a 6x6 box around px is valid for the projection
        bool bad = false;
        for (int p = 0; p < 4; p++) {
          Vec2 ppx = px + Vec2{(p & 1) ? -3 : 3, (p & 2) ? -3 : 3};
          Vec3 ray = gtModel.unprojectNoChecks(ppx);

          Vec2 ppx2;
          bool projOk = gtModel.project(ray, ppx2);
          if (!projOk || (ppx - ppx2).squaredNorm() > 1e-6) {
            bad = true;
            break;
          }
        }
        if (bad) {
          continue;
        }

        Vec3 ray = gtModel.unprojectNoChecks(px).normalized();
        for (double dist : distances) {
          Vec3 varCamPt = varT_Cam_BodyImu * (gtT_Cam_BodyImu.inverse() * (ray * dist));
          Vec2 px3;
          if (varCamModel.project(varCamPt, px3)) {
            double reprojError = (px - px3).norm();
            std::stringstream label;
            label << "Cam" << camIndex << ".ProjOffsetAt" << std::setfill('.') << std::setw(4)
                  << (int)(dist * 100) << "cm";
            allStats[label.str()].add(reprojError);
          }
        }
      }
    }

    nDone++;
    if (nDone % 2500 == 0) {
      XR_LOGI("CalibOffsets: {} / {}", nDone, prob_.rigCamToModelIndex().size());
    }
  }

  // imu extrinsics
  for (auto [rigIndexImuIndex, varIndex] : prob_.rigImuToExtrIndex()) {
    auto [rigIndex, imuIndex] = rigIndexImuIndex;
    const auto& gtT_Imu_BodyImu = fData_.factoryCalibration.calib
                                      .T_Imu_BodyImu[matcher_.slamImuIndexToFactoryCalib[imuIndex]];
    SE3 errAtImu = prob_.T_Imu_BodyImu_at(varIndex).var.value * gtT_Imu_BodyImu.inverse();
    std::stringstream label;
    label << "Imu" << imuIndex << ".Extr.";
    allStats[label.str() + "RotDegs"].add(radiansToDegrees(errAtImu.so3().log().norm()));
    allStats[label.str() + "TrMMs"].add(errAtImu.translation().norm() * 1000.0);
  }

#if 1
  // imu intrinsics
  for (auto [rigIndexImuIndex, varIndex] : prob_.rigImuToCalibIndex()) {
    auto [rigIndex, imuIndex] = rigIndexImuIndex;
    const auto& varCalib = prob_.imuCalib_at(varIndex).var.value;
    ImuCalibParam gtCalib = ImuCalibParam{
        .modelParams = fData_.factoryCalibration.calib
                           .imuModelParameters[matcher_.slamImuIndexToFactoryCalib[imuIndex]],
        .estOpts = varCalib.estOpts,
        .jacInd = varCalib.jacInd,
    };

    Eigen::VectorXd delta(::small_thing::VarSpec<ImuCalibParam>::getDynamicTangentDim(varCalib));
    ::small_thing::VarSpec<ImuCalibParam>::boxMinus(varCalib, gtCalib, delta);

    std::string label = fmt::format("Imu{}.Intr.", imuIndex);
    varCalib.eachNamedDeltaComponent(
        delta, [&](const std::string& name, Eigen::Ref<const Eigen::VectorXd> segment) {
          allStats[label + name].add(segment.norm());
        });
  }
#endif
}

} // namespace visual_inertial_ba
