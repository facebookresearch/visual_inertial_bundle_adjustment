/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <io/SaveDeviceTrajectory.h>
#include <io/SaveOnlineCalib.h>
#include <viba/common/Report.h>
#include <viba/common/Settings.h>
#include <viba/problem/PointRefinement.h>
#include <viba/problem/SingleSessionProblem.h>
#include <viba/single_session/Matcher.h>
#include <viba/single_session/SingleSessionAdapter.h>

#include <CLI/CLI.hpp>

using namespace small_thing;
using namespace visual_inertial_ba;
using namespace projectaria::tools;

int main(int argc, const char* argv[]) {
  std::filesystem::path filterPathIn;
  std::filesystem::path outputsDir;
  InitSettings initSettings;
  OptimizerSettings optSettings;
  int64_t rigIndexStart = -1, rigIndexEnd = -1;

  CLI::App app{"Test of loading filter data"};
  initSettings.addCommandLine(app);
  optSettings.addCommandLine(app);
  app.add_option("-i,--input-dir", filterPathIn, "input folder path")->required();
  app.add_option("-o,--output-dir", outputsDir, "Output folder path");
  app.add_option("--rig-start", rigIndexStart, "Index of starting rig in recording");
  app.add_option("--rig-end", rigIndexEnd, "Index of ending rig in recording");

  CLI11_PARSE(app, argc, argv);

  std::cout << "Loading..." << std::endl;
  SessionData fData;
  fData.load(filterPathIn, /* loadImuMeasurements = */ true);

  std::cout << "Building indices..." << std::endl;
  Matcher matcher;
  matcher.buildIndices(fData);

  Optimizer opt;

  std::cout << "Creating problem..." << std::endl;
  visual_inertial_ba::SingleSessionProblem problem(opt);
  visual_inertial_ba::SingleSessionAdapter adapter(problem, fData, matcher);

  adapter.initAllVariablesAndFactors(initSettings, rigIndexStart, rigIndexEnd, nullptr);

  // needed to compute factors in pt-refinement, histograms and optionally verifying jacobians
  adapter.updateRollingShutterData();

  // do point refinement
  refinePoints(opt);

  std::cout << "Showing histogram..." << std::endl;
  problem.showHistogram(false);

  if (!optSettings.dontOptimize) {
    // register points as elimination range (Schur complement trick)
    problem.registerPointVariables();
    opt.registeredVariablesToEliminationRange(); // eliminate points

    std::cout << fmt::format(
        "Optimizing:\n"
        "tot recording.: {}\n"
        "#rigs.........: {}\n"
        "#point tracks.: {}\n"
        "#cameras......: {}\n"
        "#extrinsics...: {}\n"
        "#imu calibs...: {}\n",
        microsecondsString(adapter.usedRecordingLengthUs()),
        problem.numRigVariables(),
        problem.numPointTrackVariables(),
        problem.numCameraModelVariables(),
        problem.numExtrinsicsVariables(),
        problem.numImuCalibVariables());

    auto settings = optSettings.getSettings(problem.numRigVariables());
    settings.preStepCallback = [&](int /* nIt */) {
      if (initSettings.recomputedPreIntegrations) {
        const int64_t nRecomp = adapter.regenerateAllPreintegrationsFromImuMeasurements();
        std::cout << fmt::format("Re-computed {} preintegrations", nRecomp) << std::endl;
      }
      adapter.updateRollingShutterData();
    };
    auto summary = opt.optimize(settings);
    if (optSettings.triggerDebuggingOfNonlinearities > 0) {
      std::cout << "Exiting... happy debugging!" << std::endl;
      return 0;
    }

    if (!initSettings.jsonReport.empty()) {
      writeJsonReport(initSettings.jsonReport, summary);
    }

    std::cout << "Showing histogram..." << std::endl;
    problem.showHistogram(initSettings.simpleStats);
  }

  if (initSettings.evalCalibVsFactory) {
    std::map<std::string, StatsValueContainer> allStats;
    adapter.compareCalibrationVsFactory(allStats);
    SingleSessionAdapter::showCalibrationCompResults(allStats);
  }

  if (!outputsDir.empty()) {
    std::filesystem::create_directories(outputsDir);

    saveOnlineCalib(problem, fData, outputsDir / "online_calibration.jsonl");

    saveOpenLoopTrajectory(problem, fData, outputsDir / "open_loop_framerate_trajectory.csv");

    saveCloseLoopTrajectory(problem, fData, outputsDir / "closed_loop_framerate_trajectory.csv");
  }

  return 0;
}
