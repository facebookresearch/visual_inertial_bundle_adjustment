/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
#include <small_thing/Print.h>
#include <viba/common/Enumerate.h>
#include <viba/common/Histogram.h>
#include <viba/common/StatsValueContainer.h>
#include <viba/problem/Histograms.h>
#include <viba/problem/Types.h>

#include <logging/Checks.h>
#define DEFAULT_LOG_CHANNEL "ViBa::Histograms"
#include <logging/Log.h>

namespace visual_inertial_ba {

namespace {
constexpr char TERM_COL_RESET[] = "\033[0m";
constexpr const char* TERM_COLORS[] = {
    "\033[31m", // Red
    "\033[32m", // Green
    "\033[33m", // Yellow
    "\033[34m", // Blue
    "\033[35m", // Magenta
    "\033[36m", // Cyan
};

const char* startCol(Histograms::Color c) {
  XR_CHECK_LE((int)c, (int)Histograms::ColorsEnd);
  return c == Histograms::None ? "" : TERM_COLORS[(int)c - (int)Histograms::ColorsStart];
}

const char* endCol(Histograms::Color c) {
  return c == Histograms::None ? "" : TERM_COL_RESET;
}

} // namespace

using small_thing::indent;

void Histograms::show() const {
  // visual factors can be of different types (if using different camera models)
  std::vector<small_thing::FactorStoreBase*> allVisualFactors;
  small_thing::FactorStoreBase* baseMapVisualFactors = nullptr;
  small_thing::FactorStoreBase* inertialFactors = nullptr;
  std::vector<small_thing::FactorStoreBase*> allSecondaryInertialFactors;
  small_thing::FactorStoreBase* imuCalibRWFactors = nullptr;
  std::vector<small_thing::FactorStoreBase*> allCamIntrRWFactors;
  small_thing::FactorStoreBase* imuExtrRWFactors = nullptr;
  small_thing::FactorStoreBase* camExtrRWFactors = nullptr;
  std::vector<small_thing::FactorStoreBase*> omegaPriorFactors;
  small_thing::FactorStoreBase* imuCalibFCFactors = nullptr;
  std::vector<small_thing::FactorStoreBase*> allCamIntrFCFactors;
  small_thing::FactorStoreBase* imuExtrFCFactors = nullptr;
  small_thing::FactorStoreBase* camExtrFCFactors = nullptr;

  // `opt.factorStores` are opaque classes (`FactorStoreBase`) containing an instance of
  // `FactorStore<...>`, which can give us a prettified `typeid(F).name()` where F is the function
  // type specified in `addFactor(F&& f,...)`; those `factorStores` group together the factors of
  // the same type. Our costs are lambdas, and they contain the enclosing function in their
  // prettified type name, so we search string bits to get the relevant bucket of cost terms.
  for (const auto& [tid, factorStore] : opt.factorStores.stores) {
    std::string name = factorStore->name();
    if (name.find("visual_inertial_ba::VisualFactor") != std::string::npos ||
        name.find("visual_inertial_ba::RollingShutterVisualFactor") != std::string::npos) {
      allVisualFactors.push_back(factorStore.get());
    } else if (name.find("visual_inertial_ba::BaseMapVisualFactor") != std::string::npos) {
      baseMapVisualFactors = factorStore.get();
    } else if (name.find("visual_inertial_ba::InertialFactor") != std::string::npos) {
      inertialFactors = factorStore.get();
    } else if (name.find("visual_inertial_ba::SecondaryImuInertialFactor") != std::string::npos) {
      allSecondaryInertialFactors.push_back(factorStore.get());
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addImuCalibRWFactor(") !=
        std::string::npos) {
      imuCalibRWFactors = factorStore.get();
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addCamIntrRWFactor(") !=
        std::string::npos) {
      allCamIntrRWFactors.push_back(factorStore.get());
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addImuExtrRWFactor(") !=
        std::string::npos) {
      imuExtrRWFactors = factorStore.get();
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addCamExtrRWFactor(") !=
        std::string::npos) {
      camExtrRWFactors = factorStore.get();
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addOmegaPriorFactor(") !=
        std::string::npos) {
      omegaPriorFactors.push_back(factorStore.get());
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addImuPrior(") != std::string::npos) {
      imuCalibFCFactors = factorStore.get();
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addCamIntrinsicsPrior(") !=
        std::string::npos) {
      allCamIntrFCFactors.push_back(factorStore.get());
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addImuExtrinsicsPrior(") !=
        std::string::npos) {
      imuExtrFCFactors = factorStore.get();
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addCamExtrinsicsPrior(") !=
        std::string::npos) {
      camExtrFCFactors = factorStore.get();
    }
  }

  if (!allVisualFactors.empty() && showVisual) {
    showHistogramsVisual(allVisualFactors, baseMapVisualFactors);
  }

  if (inertialFactors && showInertial) {
    showHistogramsInertial(inertialFactors, allSecondaryInertialFactors);
  }

  if (showRandomWalks) {
    showHistogramsCalib(
        false, imuCalibRWFactors, allCamIntrRWFactors, imuExtrRWFactors, camExtrRWFactors);
  }

  if (showOmegaPriors) {
    showHistogramsOmegaPriors(omegaPriorFactors);
  }

  if (showFactoryCalibPriors) {
    showHistogramsCalib(
        true, imuCalibFCFactors, allCamIntrFCFactors, imuExtrFCFactors, camExtrFCFactors);
  }
}

void Histograms::showHistogramsOmegaPriors(
    std::vector<small_thing::FactorStoreBase*> allOmegaPriorFactors) const {
  if (allOmegaPriorFactors.empty()) {
    return;
  }
  size_t numOmegaHistograms = omegaPriors.groupLabel.size();
  std::vector<Histogram> omegaPriorHistograms;
  std::vector<int64_t> numOmegaFactors(numOmegaHistograms, 0);
  std::vector<double> costContributions(numOmegaHistograms, 0.0);
  for (int i = 0; i < numOmegaHistograms; i++) {
    omegaPriorHistograms.emplace_back(0.0, 2.4, 8);
  }

  for (auto omegaPriorFactors : allOmegaPriorFactors) {
    for (int64_t count = omegaPriorFactors->numCosts(), i = 0; i < count; i++) {
      int histIndex =
          omegaPriors.factorToGroup ? omegaPriors.factorToGroup(omegaPriorFactors, i) : 0;
      XR_CHECK_GE(histIndex, 0);
      XR_CHECK_LT(histIndex, numOmegaHistograms);

      numOmegaFactors[histIndex]++;

      double covError = std::sqrt(omegaPriorFactors->covWeightedSquaredError(i) * 2.0);
      omegaPriorHistograms[histIndex].stat(covError);

      costContributions[histIndex] += omegaPriorFactors->singleCost(i).value();
    }
  }

  for (int i = 0; i < numOmegaHistograms; i++) {
    if (!numOmegaFactors[i]) {
      continue;
    }
    Color col = omegaPriors.groupCol(i);
    XR_LOGI(
        "{}{}omega priors\n"
        "cost contrib: {:.3e}, n. factors: {}  -~oOo~-  histogram (covariance-weighted costs):\n{}{}",
        startCol(col),
        omegaPriors.groupLabel[i],
        costContributions[i],
        numOmegaFactors[i],
        omegaPriorHistograms[i].show({.precision = 1}),
        endCol(col));
  }
}

void Histograms::showHistogramsVisual(
    const std::vector<small_thing::FactorStoreBase*>& allVisualFactors,
    small_thing::FactorStoreBase* baseMapVisualFactors) const {
  size_t numVisualHistograms = visual.groupLabel.size();
  std::vector<Histogram> pixelVisualFactorsHistograms, covVisualFactorsHistograms;
  std::vector<int64_t> numVisualFactors(numVisualHistograms, 0);
  std::vector<int64_t> numFailingFactors(numVisualHistograms, 0);
  std::vector<double> costContributions(numVisualHistograms, 0.0);
  for (int i = 0; i < numVisualHistograms; i++) {
    pixelVisualFactorsHistograms.emplace_back(0.0, 5.0, 5);
    pixelVisualFactorsHistograms.back().split_bucket(1, 2);
    pixelVisualFactorsHistograms.back().split_bucket(0, 2);
    covVisualFactorsHistograms.emplace_back(0.0, 2.4, 8);
  }

  for (auto visualFactors : allVisualFactors) {
    for (int64_t count = visualFactors->numCosts(), i = 0; i < count; i++) {
      int histIndex = visual.factorToGroup ? visual.factorToGroup(visualFactors, i) : 0;
      XR_CHECK_GE(histIndex, 0);
      XR_CHECK_LT(histIndex, numVisualHistograms);

      numVisualFactors[histIndex]++;

      auto cost = visualFactors->singleCost(i);
      if (!cost.has_value()) {
        numFailingFactors[histIndex]++;
        continue;
      }

      double pixelDistance = std::sqrt(visualFactors->unweightedSquaredError(i) * 2.0);
      pixelVisualFactorsHistograms[histIndex].stat(pixelDistance);

      double covError = std::sqrt(visualFactors->covWeightedSquaredError(i) * 2.0);
      covVisualFactorsHistograms[histIndex].stat(covError);

      costContributions[histIndex] += cost.value();
    }
  }

  for (int i = 0; i < numVisualHistograms; i++) {
    if (numVisualFactors[i] == 0) {
      continue;
    }

    Color col = visual.groupCol(i);
    if (!showPixelErrors) {
      XR_LOGI(
          "{}{}visual factors\n"
          "cost contrib: {:.3e}, n. factors: {}, n. failing: {}  -~oOo~-  histogram (covariance-weighted costs):\n{}{}",
          startCol(col),
          visual.groupLabel[i],
          costContributions[i],
          numVisualFactors[i],
          numFailingFactors[i],
          covVisualFactorsHistograms[i].show({.precision = 1}),
          endCol(col));
    } else {
      XR_LOGI(
          "{}{}visual factors\n"
          "cost contrib: {:.3e}, n. factors: {}, n. failing: {}  -~oOo~-  histogram (covariance-weighted costs):\n{}"
          "image-reprojection error histogram (pixel distance):\n{}{}\n",
          startCol(col),
          visual.groupLabel[i],
          costContributions[i],
          numVisualFactors[i],
          numFailingFactors[i],
          covVisualFactorsHistograms[i].show({.precision = 1}),
          indent(pixelVisualFactorsHistograms[i].show({.precision = 1})),
          endCol(col));
    }
  }

  if (baseMapVisualFactors) {
    Histogram pixelFactorsHistogram(0.0, 5.0, 5), covFactorsHistogram(0.0, 2.4, 8);
    int64_t numFactors = baseMapVisualFactors->numCosts();
    int64_t numFailing = 0.0;
    double costContrib = 0.0;

    pixelFactorsHistogram.split_bucket(1, 2);
    pixelFactorsHistogram.split_bucket(0, 2);

    for (int64_t count = baseMapVisualFactors->numCosts(), i = 0; i < count; i++) {
      auto cost = baseMapVisualFactors->singleCost(i);
      if (!cost.has_value()) {
        numFailing++;
        continue;
      }

      double pixelDistance = std::sqrt(baseMapVisualFactors->unweightedSquaredError(i) * 2.0);
      pixelFactorsHistogram.stat(pixelDistance);

      double covError = std::sqrt(baseMapVisualFactors->covWeightedSquaredError(i) * 2.0);
      covFactorsHistogram.stat(covError);

      costContrib += cost.value();
    }

    Color col = baseMap.groupCol(0);
    if (!showPixelErrors) {
      XR_LOGI(
          "{}Base map factors\n"
          "cost contrib: {:.3e}, n. factors: {}, n. failing: {}  -~oOo~-  histogram (covariance-weighted costs):\n{}{}",
          startCol(col),
          costContrib,
          numFactors,
          numFailing,
          covFactorsHistogram.show({.precision = 1}),
          endCol(col));
    } else {
      XR_LOGI(
          "{}Base map factors\n"
          "cost contrib: {:.3e}, n. factors: {}, n. failing: {}  -~oOo~-  histogram (covariance-weighted costs):\n{}"
          "image-reprojection error histogram (pixel distance):\n{}{}\n",
          startCol(col),
          costContrib,
          numFactors,
          numFailing,
          covFactorsHistogram.show({.precision = 1}),
          indent(pixelFactorsHistogram.show({.precision = 1})),
          endCol(col));
    }
  }
}

void Histograms::showHistogramsInertial(
    small_thing::FactorStoreBase* inertialFactors,
    std::vector<small_thing::FactorStoreBase*> allSecondaryInertialFactors) const {
  XR_CHECK(inertial.groupLabel.size() == secondaryInertial.groupLabel.size());
  struct Item {
    std::vector<small_thing::FactorStoreBase*> allFactors;
    std::vector<FactorRefToGroupIndex> factorToGroups;
    std::string label;
    const HistSetting& setting;
  };
  std::vector<Item> items = {
      {.allFactors = {inertialFactors},
       .factorToGroups = {inertial.factorToGroup},
       .label = "main inertial",
       .setting = inertial},
      {.allFactors = allSecondaryInertialFactors,
       .factorToGroups = std::vector<FactorRefToGroupIndex>(
           allSecondaryInertialFactors.size(), secondaryInertial.factorToGroup),
       .label = "secondary inertial",
       .setting = secondaryInertial},
  };
  if (!separateSecondaryInertial) {
    Item it = {.label = "all inertial", .setting = inertial};
    for (const auto& itx : items) {
      XR_CHECK_EQ(it.setting.groupLabel.size(), itx.setting.groupLabel.size());
      it.allFactors.insert(it.allFactors.end(), itx.allFactors.begin(), itx.allFactors.end());
      it.factorToGroups.insert(
          it.factorToGroups.end(), itx.factorToGroups.begin(), itx.factorToGroups.end());
    }
    items = std::vector<Item>{it};
  }

  for (const auto& item : items) {
    size_t numInertialHistograms = item.setting.groupLabel.size();
    std::vector<Histogram> inertialFactorsHistograms;
    std::vector<int64_t> numInertialFactors(numInertialHistograms, 0);
    std::vector<double> costContributions(numInertialHistograms, 0.0);
    for (int i = 0; i < numInertialHistograms; i++) {
      inertialFactorsHistograms.emplace_back(0.0, 2.4, 8);
    }

    // data for split rot/vel/pos histograms
    std::vector<StatsValueContainer> rotErrStats, velErrStats, posErrStats;
    std::vector<Histogram> rotErrHistograms, velErrHistograms, posErrHistograms;
    if (showRotVelPos) {
      rotErrStats.resize(numInertialHistograms);
      velErrStats.resize(numInertialHistograms);
      posErrStats.resize(numInertialHistograms);
      for (int i = 0; i < numInertialHistograms; i++) {
        rotErrHistograms.emplace_back(0.0, 0.1, 5); // degrees
        velErrHistograms.emplace_back(0.0, 0.5, 5); // cm/s
        posErrHistograms.emplace_back(0.0, 1.0, 5); // cm
      }
    }

    for (const auto& [j, factorStore] : enumerate(item.allFactors)) {
      if (!factorStore) {
        continue;
      }
      for (int64_t count = factorStore->numCosts(), i = 0; i < count; i++) {
        int histIndex = item.factorToGroups[j] ? item.factorToGroups[j](factorStore, i) : 0;
        XR_CHECK_GE(histIndex, 0);
        XR_CHECK_LT(histIndex, numInertialHistograms);

        if (showRotVelPos) {
          Vec9 rotVelPosE;
          factorStore->unweightedError(i, rotVelPosE);
          double rotErrDegrees = radiansToDegrees(rotVelPosE.head<3>().norm());
          double velErrCmS = rotVelPosE.segment<3>(3).norm() * 100.0;
          double posErrCm = rotVelPosE.tail<3>().norm() * 100.0;
          rotErrStats[histIndex].add(rotErrDegrees);
          velErrStats[histIndex].add(velErrCmS);
          posErrStats[histIndex].add(posErrCm);
          rotErrHistograms[histIndex].stat(rotErrDegrees);
          velErrHistograms[histIndex].stat(velErrCmS);
          posErrHistograms[histIndex].stat(posErrCm);
        }

        numInertialFactors[histIndex]++;
        double totalInertialError = std::sqrt(factorStore->covWeightedSquaredError(i) * 2.0);
        inertialFactorsHistograms[histIndex].stat(totalInertialError);

        costContributions[histIndex] += factorStore->singleCost(i).value();
      }
    }

    for (int i = 0; i < numInertialHistograms; i++) {
      if (numInertialFactors[i] == 0) {
        continue;
      }

      Color col = item.setting.groupCol(i);

      if (!showRotVelPos) {
        XR_LOGI(
            "{}{}{} factors\n"
            "cost contrib: {:.3e}, n. factors: {}  -~oOo~-  histogram (covariance-weighted costs):\n{}{}",
            startCol(col),
            item.setting.groupLabel[i],
            item.label,
            costContributions[i],
            numInertialFactors[i],
            inertialFactorsHistograms[i].show({.precision = 1}),
            endCol(col));
      } else {
        XR_LOGI(
            "{}{}{} factors\n"
            "cost contrib: {:.3e}, n. factors: {}  -~oOo~-  histogram (covariance-weighted costs):\n{}"
            "Rotation p50: {:.3g}\u00BA, MAE: {:.3g}\u00BA, RMSE: {:.3g}\u00BA  -~oOo~-   histogram (degrees):\n{}\n"
            "Velocity p50: {:.3g}cm/s, MAE: {:.3g}cm/s, RMSE: {:.3g}cm/s  -~oOo~-  histogram (cm/s):\n{}\n"
            "Position p50: {:.3g}cm, MAE: {:.3g}cm, RMSE: {:.3g}cm  -~oOo~-  histogram (cm):\n{}{}\n",
            startCol(col),
            item.setting.groupLabel[i],
            item.label,
            costContributions[i],
            numInertialFactors[i],
            inertialFactorsHistograms[i].show({.precision = 1}),
            rotErrStats[i].p50(),
            rotErrStats[i].mean(),
            rotErrStats[i].rmse(),
            indent(rotErrHistograms[i].show({.precision = 2})),
            velErrStats[i].p50(),
            velErrStats[i].mean(),
            velErrStats[i].rmse(),
            indent(velErrHistograms[i].show({.precision = 1})),
            posErrStats[i].p50(),
            posErrStats[i].mean(),
            posErrStats[i].rmse(),
            indent(posErrHistograms[i].show({.precision = 1})),
            endCol(col));
      }
    }
  }
}

void Histograms::showHistogramsCalib(
    bool isFactoryCalibPriors,
    small_thing::FactorStoreBase* imuCalibRWFactors,
    std::vector<small_thing::FactorStoreBase*> allCamIntrRWFactors,
    small_thing::FactorStoreBase* imuExtrRWFactors,
    small_thing::FactorStoreBase* camExtrRWFactors) const {
  struct Item {
    std::vector<small_thing::FactorStoreBase*> allFactors;
    std::vector<FactorRefToGroupIndex> factorToGroups;
    std::string label;
    const HistSetting& setting;
  };
  std::vector<Item> items = {
      {.allFactors = {imuCalibRWFactors},
       .factorToGroups = {rwImuCalib.factorToGroup},
       .label = "IMU calibration",
       .setting = isFactoryCalibPriors ? fpCamIntr : rwImuCalib},
      {.allFactors = allCamIntrRWFactors,
       .factorToGroups =
           std::vector<FactorRefToGroupIndex>(allCamIntrRWFactors.size(), rwCamIntr.factorToGroup),
       .label = "Cam intrinsics",
       .setting = isFactoryCalibPriors ? fpCamIntr : rwCamIntr},
      {.allFactors = {imuExtrRWFactors},
       .factorToGroups = {rwImuExtr.factorToGroup},
       .label = "IMU extrinsics",
       .setting = isFactoryCalibPriors ? fpImuExtr : rwImuExtr},
      {.allFactors = {camExtrRWFactors},
       .factorToGroups = {rwCamExtr.factorToGroup},
       .label = "Cam extrinsics",
       .setting = isFactoryCalibPriors ? fpCamExtr : rwCamExtr},
  };
  if (showAggregateCalibFactors) {
    Item it = {.label = "aggregate", .setting = isFactoryCalibPriors ? fpImuCalib : rwImuCalib};
    for (const auto& itx : items) {
      XR_CHECK_EQ(it.setting.groupLabel.size(), itx.setting.groupLabel.size());
      it.allFactors.insert(it.allFactors.end(), itx.allFactors.begin(), itx.allFactors.end());
      it.factorToGroups.insert(
          it.factorToGroups.end(), itx.factorToGroups.begin(), itx.factorToGroups.end());
    }
    items = std::vector<Item>{it};
  }

  for (const auto& item : items) {
    if (item.allFactors.empty()) {
      continue;
    }

    size_t numCFHistograms = item.setting.groupLabel.size();
    std::vector<Histogram> calibFactorsHistograms;
    std::vector<int64_t> numCalibFactors(numCFHistograms, 0);
    std::vector<double> costContributions(numCFHistograms, 0.0);
    for (int i = 0; i < numCFHistograms; i++) {
      calibFactorsHistograms.emplace_back(0.0, 2.4, 8);
    }

    for (auto [q, factorStore] : enumerate(item.allFactors)) {
      if (!factorStore) {
        continue;
      }

      for (int64_t count = factorStore->numCosts(), i = 0; i < count; i++) {
        double error = std::sqrt(factorStore->covWeightedSquaredError(i) * 2.0);
        int histIndex = item.factorToGroups[q] ? item.factorToGroups[q](factorStore, i) : 0;
        XR_CHECK_GE(histIndex, 0);
        XR_CHECK_LT(histIndex, numCFHistograms);

        numCalibFactors[histIndex]++;
        calibFactorsHistograms[histIndex].stat(error);
        costContributions[histIndex] += factorStore->singleCost(i).value();
      }
    }

    for (int i = 0; i < numCFHistograms; i++) {
      if (numCalibFactors[i] == 0) {
        continue;
      }

      Color col = item.setting.groupCol(i);
      XR_LOGI(
          "{}{}{} {} factors\n"
          "cost contrib: {:.3e}, n. factors: {}  -~oOo~-  histogram (covariance-weighted cost):\n{}{}",
          startCol(col),
          item.setting.groupLabel[i],
          item.label,
          isFactoryCalibPriors ? "factory-calib prior" : "random-walk",
          costContributions[i],
          numCalibFactors[i],
          calibFactorsHistograms[i].show({.precision = 1}),
          endCol(col));
    }
  }
}

} // namespace visual_inertial_ba
