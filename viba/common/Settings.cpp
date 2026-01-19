/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <viba/common/Settings.h>
#include <exception>
#include <optional>

#include <logging/Checks.h>
#define DEFAULT_LOG_CHANNEL "ViBa::Settings"
#include <logging/Log.h>

namespace visual_inertial_ba {

constexpr char CalibInitArgSpec[] =
    "imu-calib|imu-extr|imu-all|cam-intr|cam-extr|cam-all|all-extr|all";
constexpr char TrajectoryInitArgSpec[] = "pose|vel|omega|all";
constexpr char ImuEstArgSpec[] =
    "gyro-bias|accel-bias|gyro-scale|accel-scale|gyro-nonorth|accel-nonorth|\n"
    "  gyro-g-sensitivity|accel-axes-offset|gyro-nonlinearity|accel-nonlinearity|\n"
    "  reference-imu-time-offset|gyro-accel-time-offset|all|all-but-time-offsets|\n"
    "  all-but-biases|all-but-biases-and-scales|all-non-orths|all-time-offsets";

void InitSettings::addCalibOverride(CLI::App& app) {
  app.add_option(
         "--calib-factory",
         calibFactory,
         "Set calib set to (constant) factory values. One or more comma-sep of:\n  " +
             std::string(CalibInitArgSpec))
      ->group("Calibration");
  app.add_flag_function(
         "--no-fprio",
         [&](size_t /* something */) {
           imuFactorCalibInflate = camIntrFactorCalibInflate = imuExtrFactorCalibInflate =
               camExtrFactorCalibInflate = 0.0;
         },
         "Disable all factor calibration priors")
      ->group("Calibration");
  app.add_flag(
         "--estimate-readout-time",
         estimateReadoutTime,
         "Estimate readout-time for rolling shutter cameras")
      ->group("Calibration");
  app.add_flag(
         "--estimate-time-offset",
         estimateTimeOffset,
         "Estimate time-offset (only for rolling shutter cameras ATM)")
      ->group("Calibration");
  app.add_flag(
         "--recompute-preint",
         recomputedPreIntegrations,
         "Recompute pre-integration, at every optimization step")
      ->group("Calibration");
  app.add_option(
         "--imu-lrad",
         imuLossRadius,
         fmt::format("Radius for IMU factor loss (default: {:.02f})", kImuErrorHuberLossWidth))
      ->group("Huber Soft Loss");
  app.add_option(
         "--imu-lcut",
         imuLossCutoff,
         fmt::format("Cutoff for IMU factor loss (default: {:.02f})", kImuErrorHuberLossCutoff))
      ->group("Huber Soft Loss");
}

void InitSettings::addCommandLine(CLI::App& app) {
  app.add_option(
         "--imu-calib-rw-infl",
         imuRWinflate,
         "Inflate factor for IMU calibration random walk covariances")
      ->group("Factor Weighting");
  app.add_option(
         "--cam-intr-rw-infl",
         camIntrRWinflate,
         "Inflate factor for Cam intrinsics random walk covariances")
      ->group("Factor Weighting");
  app.add_option(
         "--imu-extr-rw-infl",
         imuExtrRWinflate,
         "Inflate factor for IMU extrinsics random walk covariances")
      ->group("Factor Weighting");
  app.add_option(
         "--cam-extr-rw-infl",
         camExtrRWinflate,
         "Inflate factor for Cam extrinsics random walk covariances")
      ->group("Factor Weighting");

  app.add_option(
         "--imu-calib-fprio-infl",
         imuFactorCalibInflate,
         "Inflate factor for IMU calibration factor calibration prior (default: 100.0)")
      ->group("Factor Weighting");
  app.add_option(
         "--cam-intr-fprio-infl",
         camIntrFactorCalibInflate,
         "Inflate factor for Cam intrinsics factor calibration prior (default: 100.0)")
      ->group("Factor Weighting");
  app.add_option(
         "--imu-extr-fprio-infl",
         imuExtrFactorCalibInflate,
         "Inflate factor for IMU extrinsics factor calibration prior (default: 100.0)")
      ->group("Factor Weighting");
  app.add_option(
         "--cam-extr-fprio-infl",
         camExtrFactorCalibInflate,
         "Inflate factor for Cam extrinsics factor calibration prior (default: 100.0)")
      ->group("Factor Weighting");
  app.add_flag_function(
         "--no-fprio",
         [&](size_t /* something */) {
           imuFactorCalibInflate = camIntrFactorCalibInflate = imuExtrFactorCalibInflate =
               camExtrFactorCalibInflate = 0.0;
         },
         "Disable all factor calibration priors")
      ->group("Factor Weighting");

  app.add_option(
         "--tracking-obs-lrad",
         trackingObsLossRadius,
         fmt::format(
             "Radius for tracking observation loss (default: {:.02f})",
             kReprojectionErrorHuberLossWidth))
      ->group("Huber Soft Loss");
  app.add_option(
         "--tracking-obs-lcut",
         trackingObsLossCutoff,
         fmt::format(
             "Cutoff for tracking observation loss (default: {:.02f})",
             kReprojectionErrorHuberLossCutoff))
      ->group("Huber Soft Loss");
  app.add_option(
         "--imu-lrad",
         imuLossRadius,
         fmt::format("Radius for IMU factor loss (default: {:.02f})", kImuErrorHuberLossWidth))
      ->group("Huber Soft Loss");
  app.add_option(
         "--imu-lcut",
         imuLossCutoff,
         fmt::format("Cutoff for IMU factor loss (default: {:.02f})", kImuErrorHuberLossCutoff))
      ->group("Huber Soft Loss");
  app.add_option(
         "--calib-constant",
         calibConstant,
         "Set to constant. One or more comma-sep of:\n  " + std::string(CalibInitArgSpec))
      ->group("Calibration");
  app.add_option(
         "--calib-factory",
         calibFactory,
         "Set calib set to (constant) factory values. One or more comma-sep of:\n  " +
             std::string(CalibInitArgSpec))
      ->group("Calibration");
  app.add_option(
         "--imu-calib-estimation-options",
         imuCalibEstimationOptions,
         "Set individual IMU calib fields to (constant) override values. Comma-sep of:\n  " +
             std::string(ImuEstArgSpec))
      ->group("Calibration");
  app.add_option(
         "--imu-calib-fields-to-factory",
         imuCalibFieldsToFactory,
         "Set individual IMU calib fields to (constant) override values. Comma-sep of:\n  " +
             std::string(ImuEstArgSpec))
      ->group("Calibration");
  app.add_flag(
         "--eval-calib-vs-factory",
         evalCalibVsFactory,
         "Evaluate deviation of calibration parameters wrt factory calibration")
      ->group("Calibration");
  app.add_flag(
         "--estimate-readout-time",
         estimateReadoutTime,
         "Estimate readout-time for rolling shutter cameras")
      ->group("Calibration");
  app.add_flag(
         "--estimate-time-offset",
         estimateTimeOffset,
         "Estimate time-offset (only for rolling shutter cameras ATM)")
      ->group("Calibration");
  app.add_flag(
         "--recompute-preint",
         recomputedPreIntegrations,
         "Recompute pre-integration, at every optimization step")
      ->group("Calibration");

  app.add_option(
         "--trajectory-constant",
         trajectoryConstant,
         "Set trajectory components to constants. Comma-sep of: " +
             std::string(TrajectoryInitArgSpec))
      ->group("Trajectory");
  app.add_option(
         "--trajectory-to-gt",
         trajectoryToGt,
         "Set trajectory components to loading from GT. Comma-sep of: " +
             std::string(TrajectoryInitArgSpec))
      ->group("Trajectory");
  app.add_option(
         "--gt-trajectory-base-name",
         gtTrajectoryBaseName,
         "'base' filename of pre-downloaded trajectory GT file (inside session data dir)")
      ->group("Trajectory");
  app.add_option(
         "--gt-trajectory-type",
         gtTrajectoryType,
         "Type of trajectory GT, as defined in MapGtReader.cpp (default: frl_synthetic_spline)\n"
         "NOTE: must support querying arbitrary timestamps, to deduce linear/angular velocities")
      ->group("Trajectory");

  app.add_flag(
         "--optimize-detector-bias",
         optimizeDetectorBias,
         "Optimize image detector bias. Note: it correlates with intrinsics principal point")
      ->group("Debugging");

  app.add_option(
      "--json-report",
      jsonReport,
      "Path for JSON output containing a report on the optimization process");
  app.add_flag("--simple-stats", simpleStats, "Simplified histogram stats");
  app.add_flag("--verbose", verbose, "verbose: print residuals");

  app.add_flag("--no-rgb", noRgb, "Map KeyRigs were not constructed using RGB images");
}

void T3InitSettings::addCommandLine(CLI::App& app) {
  InitSettings::addCommandLine(app);

  app.add_flag(
      "--const-base-map",
      constBaseMap,
      "Map data used as constant base map");

  app.add_option(
         "--t3map-obs-infl",
         t3MapObsCovInflate,
         "Inflate factor for covariances of T3Map's map point's observations")
      ->group("Factor Weighting");
  app.add_option(
         "--t3map-obs-lrad",
         t3MapObsLossRadius,
         "Radius for T3Map's map point's observations (default: " +
             std::to_string(kReprojectionErrorHuberLossWidth) + ")")
      ->group("Huber Soft Loss");
  app.add_option(
         "--t3map-obs-lcut",
         t3MapObsLossCutoff,
         "Cutoff for T3Map's map point's observations (default: " +
             std::to_string(kReprojectionErrorHuberLossCutoff) + ")")
      ->group("Huber Soft Loss");
}

using namespace small_thing;

static std::unordered_map<std::string, std::optional<Optimizer::SolverType>> argToSolver{
    {"direct", Optimizer::Solver_Direct},
    {"gauss-seidel", Optimizer::Solver_PCG_GaussSeidel},
    {"jacobi", Optimizer::Solver_PCG_Jacobi},
    {"auto", {}},
};
static const char* solverArgs = "auto|direct|gauss-seidel|jacobi";

void OptimizerSettings::addCommandLine(CLI::App& app, const std::string& group, bool skipAdvanced) {
  app.add_option(
         "--max-num-iterations", maxNumIterations, "Max number of iterations (default: 250)")
      ->group(group);
  app.add_option(
         "--linear-solver", linearSolverArg, "Linear solver (" + std::string(solverArgs) + ")")
      ->group(group);
  app.add_option(
         "--pcg-max-iterations",
         pcgMaxIterations,
         "Max iterations for PCG iterative solvers (Gauss-Seidel or Jacobi), default: 40")
      ->group(group);
  app.add_flag("--dont-optimize", dontOptimize, "do not optimize (to just print histograms, etc)")
      ->group(group);
  app.add_option("--num-threads", numThreads, "Number of threads to use for optimization")
      ->group(group);
  if (!skipAdvanced) {
    app.add_option(
           "--debug-nonlinearities-at",
           triggerDebuggingOfNonlinearities,
           "Trigger debugging of factor non linearities at iteration, default: -1 (ie disabled)")
        ->group(group);
  }
  app.callback([&]() {
    XR_CHECK(
        argToSolver.count(linearSolverArg), "Linear solver option must be one of: {}", solverArgs);
  });
}

Optimizer::Settings OptimizerSettings::getSettings(int numRigs, bool muted) const {
  auto solverIt = argToSolver.find(linearSolverArg);
  XR_CHECK(solverIt != argToSolver.end(), "Linear solver option must be one of: {}", solverArgs);
  auto linearSolver = solverIt->second;

  if (!linearSolver.has_value()) {
    linearSolver = numRigs >= kNumRigsForIterative ? Optimizer::Solver_PCG_GaussSeidel
                                                   : Optimizer::Solver_Direct;
    if (!muted) {
      XR_LOGI(
          "Linear solver automatically set to '{}' (numRigs: {}, threshold for iterative: {})",
          Optimizer::solverToString(*linearSolver),
          numRigs,
          kNumRigsForIterative);
    }
  }

  return {
      .maxNumIterations = maxNumIterations,
      .numThreads = numThreads,
      .solverType = *linearSolver,
      .pcgMaxIterations = pcgMaxIterations,
      .triggerDebuggingOfNonlinearities = triggerDebuggingOfNonlinearities,
  };
}

} // namespace visual_inertial_ba
