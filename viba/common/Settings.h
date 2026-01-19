/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <small_thing/Optimizer.h>
#include <viba/common/Constants.h>
#include <filesystem>
#include <vector>

namespace CLI {
class App;
} // namespace CLI

namespace visual_inertial_ba {

struct InitSettings {
  void addCommandLine(CLI::App& app);

  void addCalibOverride(CLI::App& app);

  std::string calibConstant;
  std::string calibFactory;
  std::string imuCalibEstimationOptions;
  std::string imuCalibFieldsToFactory;
  bool estimateReadoutTime = false;
  bool estimateTimeOffset = false;
  bool evalCalibVsFactory = false;

  std::string trajectoryConstant;
  std::string trajectoryToGt;
  std::string gtTrajectoryBaseName;
  std::string gtTrajectoryType = "frl_synthetic_spline";

  std::filesystem::path jsonReport;

  double trackingObsLossRadius = kReprojectionErrorHuberLossWidth;
  double trackingObsLossCutoff = kReprojectionErrorHuberLossCutoff;

  double imuRWinflate = 1.0;
  double camIntrRWinflate = 1.0;
  double imuExtrRWinflate = 1.0;
  double camExtrRWinflate = 1.0;

  double imuFactorCalibInflate = 100.0;
  double camIntrFactorCalibInflate = 100.0;
  double imuExtrFactorCalibInflate = 100.0;
  double camExtrFactorCalibInflate = 100.0;

  bool recomputedPreIntegrations = false;

  bool optimizeDetectorBias = false;

  bool verbose = false;
  bool simpleStats = false;

  double imuLossRadius = kImuErrorHuberLossWidth;
  double imuLossCutoff = kImuErrorHuberLossWidth;

  bool noRgb = false;
};

struct T3InitSettings : InitSettings {
  void addCommandLine(CLI::App& app);

  bool constBaseMap = false;

  double t3MapObsLossRadius = kReprojectionErrorHuberLossWidth;
  double t3MapObsLossCutoff = kReprojectionErrorHuberLossCutoff;
  double t3MapObsCovInflate = 1.0;
};

struct OptimizerSettings {
  void
  addCommandLine(CLI::App& app, const std::string& group = "Optimizer", bool skipAdvanced = false);

  small_thing::Optimizer::Settings getSettings(int numRigs, bool muted = false) const;

  int maxNumIterations = 250;
  std::string linearSolverArg = "auto";
  int pcgMaxIterations = 40;
  bool dontOptimize = false;
  unsigned int numThreads = 8;

  int triggerDebuggingOfNonlinearities = -1;
};

} // namespace visual_inertial_ba
