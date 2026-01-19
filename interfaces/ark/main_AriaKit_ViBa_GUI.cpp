/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <memory>
#include <thread>

#include "sokol_app.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"
#include "sokol_log.h"

// ImGui
#include "imgui.h"
#include "imgui_internal.h" // For DockBuilder API

#include "util/sokol_imgui.h"

#include "util/sokol_gfx_imgui.h"

// ImPlot
#include "implot.h"

#undef Success // X11 is leaking this macro

// Custom 3D Viewer
#include "gui/Geometry.h"
#include "gui/Viewer3D.h"

// Monitoring state
#include "gui/HistogramExtractor.h"
#include "gui/MonitoringState.h"

// ViBa includes (for running optimization in thread)
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
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
using namespace viba_gui;

// Global monitoring state
static std::shared_ptr<MonitoringState> g_monitoring_state;
static std::thread g_optimization_thread;
static std::unique_ptr<Viewer3D> g_3d_viewer;
static geometry::GeometryBuffer g_trajectory_buf;
static geometry::GeometryBuffer g_points_buf;

// UI state
static int g_selected_iteration = -1; // -1 means "latest"
static bool g_auto_follow_latest = true;
static bool g_show_imgui_demo = false; // ImGui demo off by default

// Sokol state
static sg_pass_action pass_action;

// Helper function to get the iteration data to display
IterationData getCurrentIterationData() {
  size_t history_size = g_monitoring_state->getHistorySize();

  if (history_size == 0) {
    return IterationData{};
  }

  // Auto-follow latest iteration
  if (g_auto_follow_latest || g_selected_iteration < 0) {
    g_selected_iteration = history_size - 1;
    return g_monitoring_state->getLatest();
  }

  // Clamp to valid range
  if (g_selected_iteration >= static_cast<int>(history_size)) {
    g_selected_iteration = history_size - 1;
  }

  return g_monitoring_state->getIteration(g_selected_iteration);
}

// Function to run optimization in background thread
void runOptimizationThread(
    std::filesystem::path filterPathIn,
    std::filesystem::path outputsDir,
    InitSettings initSettings,
    OptimizerSettings optSettings,
    int64_t rigIndexStart,
    int64_t rigIndexEnd,
    std::shared_ptr<MonitoringState> monitoringState) {
  monitoringState->setOptimizationRunning(true);

  try {
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

    // Set problem stats
    ProblemStats stats;
    stats.num_rigs = problem.numRigVariables();
    stats.num_point_tracks = problem.numPointTrackVariables();
    stats.num_cameras = problem.numCameraModelVariables();
    stats.num_extrinsics = problem.numExtrinsicsVariables();
    stats.num_imu_calibs = problem.numImuCalibVariables();
    stats.recording_length_us = adapter.usedRecordingLengthUs();
    monitoringState->setProblemStats(stats);

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

      // Track iteration data
      int iteration_counter = 0;
      auto start_time = std::chrono::high_resolution_clock::now();

      // Use log function to capture cost and other info
      std::string last_log_message;
      settings.log = [&](const std::string& msg) {
        std::cout << msg << std::endl;
        last_log_message = msg;
      };

      settings.preStepCallback = [&](int nIt) {
        auto iter_start = std::chrono::high_resolution_clock::now();

        if (initSettings.recomputedPreIntegrations) {
          const int64_t nRecomp = adapter.regenerateAllPreintegrationsFromImuMeasurements();
          std::cout << fmt::format("Re-computed {} preintegrations", nRecomp) << std::endl;
        }
        adapter.updateRollingShutterData();

        // Collect iteration data
        IterationData iter_data;
        iter_data.iteration = iteration_counter++;

        // Extract histogram data
        extractHistograms(problem, opt, iter_data.residuals_by_type);

        // Precompute statistics for histogram display and convergence tracking (avoid recomputing
        // every frame)
        for (const auto& [factor_type, residuals] : iter_data.residuals_by_type) {
          if (!residuals.empty()) {
            // Count
            iter_data.residuals_count_by_type[factor_type] = residuals.size();

            // Mean
            double sum = std::accumulate(residuals.begin(), residuals.end(), 0.0);
            iter_data.residuals_mean_by_type[factor_type] = sum / residuals.size();

            // 95th percentile
            std::vector<double> sorted_residuals = residuals;
            size_t p95_idx = static_cast<size_t>(residuals.size() * 0.95);
            if (p95_idx >= sorted_residuals.size()) {
              p95_idx = sorted_residuals.size() - 1;
            }
            std::nth_element(
                sorted_residuals.begin(),
                sorted_residuals.begin() + p95_idx,
                sorted_residuals.end());
            iter_data.residuals_p95_by_type[factor_type] = sorted_residuals[p95_idx];

            // Cost (sum of squared residuals)
            double cost = 0.0;
            for (const auto& r : residuals) {
              cost += r * r;
            }
            iter_data.cost_by_type[factor_type] = cost;
          }
        }

        // Extract trajectory (use inverse to get world position of camera)
        const auto& poses = problem.inertialPoses();
        auto sortedRigIndices = problem.sortedRigIndices();
        iter_data.camera_positions.reserve(poses.size());
        iter_data.camera_orientations.reserve(poses.size());

        for (int64_t idx : sortedRigIndices) {
          const auto& pose_vars = problem.inertialPose(idx);
          const SE3& T_bodyImu_world = pose_vars.T_bodyImu_world.value;
          // Inverse gives us T_world_bodyImu, whose translation is the body position in world
          SE3 T_world_bodyImu = T_bodyImu_world.inverse();
          iter_data.camera_positions.push_back(T_world_bodyImu.translation());
          iter_data.camera_orientations.push_back(T_world_bodyImu.unit_quaternion());
        }

        // Extract 3D points (sample to avoid too many points)
        const int64_t num_points = problem.numPointTrackVariables();
        const int64_t max_points = 5000; // Limit for visualization
        const int64_t stride = std::max<int64_t>(1, num_points / max_points);

        iter_data.point_positions.reserve(std::min(num_points, max_points));
        for (int64_t i = 0; i < num_points; i += stride) {
          const auto& pt_var = problem.pointTrack(i);
          iter_data.point_positions.push_back(pt_var.value);
        }

        auto iter_end = std::chrono::high_resolution_clock::now();
        iter_data.time_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(iter_end - iter_start);

        // Extract calibration parameters per rig (for temporal variation)
        iter_data.rig_calibrations.clear();
        iter_data.rig_calibrations.reserve(sortedRigIndices.size());

        for (int64_t rig_idx : sortedRigIndices) {
          IterationData::RigCalibration rig_calib;
          rig_calib.rig_index = rig_idx;

          // Get timestamp from session data (inertialPoses is a vector indexed by rig_idx)
          if (rig_idx >= 0 && rig_idx < static_cast<int64_t>(fData.inertialPoses.size())) {
            rig_calib.timestamp_sec =
                fData.inertialPoses[rig_idx].timestamp_us / 1e6; // Convert microseconds to seconds
          }

          // IMU calibrations (try up to 4 IMUs)
          for (int64_t imu_idx = 0; imu_idx < 4; imu_idx++) {
            try {
              const auto& imu_calib_var = problem.imuCalib(rig_idx, imu_idx);
              const auto& calib = imu_calib_var.var.value;
              rig_calib.imu_accel_bias[imu_idx] = calib.modelParams.accelBiasMSec2;
              rig_calib.imu_gyro_bias[imu_idx] = calib.modelParams.gyroBiasRadSec;
              rig_calib.imu_time_offset_accel[imu_idx] = calib.modelParams.dtReferenceAccelSec;
              rig_calib.imu_time_offset_gyro[imu_idx] = calib.modelParams.dtReferenceGyroSec;
            } catch (...) {
              break; // No more IMUs
            }
          }

          // Camera intrinsics (try up to 10 cameras)
          for (int64_t cam_idx = 0; cam_idx < 10; cam_idx++) {
            try {
              const auto& cam_model_var = problem.cameraModel(rig_idx, cam_idx);
              const auto& model = cam_model_var.var.value;
              Eigen::Vector2d focal = model.model.getFocalLengths();
              rig_calib.camera_focal_length_x[cam_idx] = focal[0];
              rig_calib.camera_focal_length_y[cam_idx] = focal[1];
              rig_calib.camera_time_offset[cam_idx] = model.timeOffsetSec_Dev_Camera();
            } catch (...) {
              break; // No more cameras
            }
          }

          // Camera-to-camera baseline distances (try all pairs up to 10 cameras)
          for (int64_t cam_i = 0; cam_i < 10; cam_i++) {
            for (int64_t cam_j = cam_i + 1; cam_j < 10; cam_j++) {
              try {
                const auto& T_Cam_i_BodyImu = problem.T_Cam_BodyImu(rig_idx, cam_i).var.value;
                const auto& T_Cam_j_BodyImu = problem.T_Cam_BodyImu(rig_idx, cam_j).var.value;
                // Compute relative transform: T_Cam_j_Cam_i = T_Cam_j_BodyImu * T_BodyImu_Cam_i
                SE3 T_Cam_j_Cam_i = T_Cam_j_BodyImu * T_Cam_i_BodyImu.inverse();
                double distance = T_Cam_j_Cam_i.translation().norm();
                rig_calib.camera_baseline_distance[{cam_i, cam_j}] = distance;
              } catch (...) {
                break; // No more camera pairs
              }
            }
          }

          iter_data.rig_calibrations.push_back(rig_calib);
        }

        // Get current cost from optimizer
        iter_data.cost = opt.computeCost();
        iter_data.cost_reduction = 0.0; // Will be computed from cost difference

        // Push to monitoring state
        monitoringState->pushIteration(iter_data);
      };

      auto summary = opt.optimize(settings);

      if (!initSettings.jsonReport.empty()) {
        writeJsonReport(initSettings.jsonReport, summary);
      }

      std::cout << "Showing histogram..." << std::endl;
      problem.showHistogram(initSettings.simpleStats);
    }

    if (!outputsDir.empty()) {
      std::filesystem::create_directories(outputsDir);
      saveOnlineCalib(problem, fData, outputsDir / "online_calibration.jsonl");
      saveOpenLoopTrajectory(problem, fData, outputsDir / "open_loop_framerate_trajectory.csv");
      saveCloseLoopTrajectory(problem, fData, outputsDir / "closed_loop_framerate_trajectory.csv");
    }

  } catch (const std::exception& e) {
    std::cerr << "Optimization failed: " << e.what() << std::endl;
  }

  monitoringState->setOptimizationRunning(false);
  monitoringState->setOptimizationDone(true);
}

void init() {
  // Setup Sokol GFX
  sg_desc desc = {};
  desc.environment = sglue_environment();
  desc.logger.func = slog_func;
  sg_setup(&desc);

  // Setup ImGui with docking enabled
  simgui_desc_t simgui_desc = {};
  simgui_desc.max_vertices = 1000000; // Increase from default 65536 to support large meshes
  simgui_setup(&simgui_desc);

  // Enable docking
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  // Setup ImPlot
  ImPlot::CreateContext();

  // Setup Sokol GFX debugger (must be after sg_setup)
  sgimgui_desc_t sgimgui_desc = {};
  sgimgui_setup(&sgimgui_desc);

  // Setup Custom 3D Viewer
  g_3d_viewer = std::make_unique<Viewer3D>();

  // Clear color
  pass_action.colors[0].load_action = SG_LOADACTION_CLEAR;
  pass_action.colors[0].clear_value = {0.15f, 0.15f, 0.15f, 1.0f};

  // Note: g_monitoring_state is initialized in sokol_main() before thread starts

  printf("Sokol + ImGui + ImPlot + ImPlot3D initialized successfully!\n");
  printf("Ready to monitor Visual Inertial Bundle Adjustment\n");
}

void renderMenuBar() {
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("Tools")) {
      // Just debug options in menu
      ImGui::Checkbox("Show ImGui Demo", &g_show_imgui_demo);
      sgimgui_draw_menu("sokol-gfx");
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }
}

void renderControlsAndStatusPanel() {
  ImGui::Begin("Controls & Status", nullptr, ImGuiWindowFlags_NoCollapse);

  // Status section
  bool is_running = g_monitoring_state->isOptimizationRunning();
  bool is_done = g_monitoring_state->isOptimizationDone();

  if (is_running) {
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "● RUNNING");
  } else if (is_done) {
    ImGui::TextColored(ImVec4(0.0f, 0.5f, 1.0f, 1.0f), "● COMPLETED");
  } else {
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "● WAITING");
  }

  auto current = getCurrentIterationData();
  ImGui::Text("Iteration: %d", current.iteration);
  ImGui::Text("Time/Iter: %lld ms", current.time_ms.count());

  ImGui::Spacing();
  ImGui::SeparatorText("Problem Statistics");

  auto stats = g_monitoring_state->getProblemStats();
  ImGui::Text("Rigs: %lld", stats.num_rigs);
  ImGui::Text("Point Tracks: %lld", stats.num_point_tracks);
  ImGui::Text("Cameras: %lld", stats.num_cameras);
  ImGui::Text("Extrinsics: %lld", stats.num_extrinsics);
  ImGui::Text("IMU Calibs: %lld", stats.num_imu_calibs);
  ImGui::Text("Recording: %.2f sec", stats.recording_length_us / 1e6);

  ImGui::Spacing();
  ImGui::SeparatorText("Iteration Control");

  size_t num_iterations = g_monitoring_state->getHistorySize();

  if (num_iterations > 0) {
    // Checkbox to auto-follow latest
    if (ImGui::Checkbox("Auto-follow Latest", &g_auto_follow_latest)) {
      if (g_auto_follow_latest) {
        g_selected_iteration = num_iterations - 1;
      }
    }

    ImGui::Spacing();

    // Slider to select iteration
    ImGui::Text("Iteration:");
    int slider_value = g_selected_iteration;
    if (ImGui::SliderInt("##iteration", &slider_value, 0, num_iterations - 1)) {
      g_selected_iteration = slider_value;
      g_auto_follow_latest = false; // Disable auto-follow when manually selecting
    }

    // Show current iteration number
    if (g_selected_iteration >= 0 && g_selected_iteration < static_cast<int>(num_iterations)) {
      ImGui::Text("Showing: %d / %d", g_selected_iteration, static_cast<int>(num_iterations) - 1);
    }
  } else {
    ImGui::TextDisabled("No iteration data available");
  }

  ImGui::End();
}

void renderConvergencePlots() {
  ImGui::Begin("Convergence", nullptr, ImGuiWindowFlags_NoCollapse);

  size_t history_size = g_monitoring_state->getHistorySize();
  if (history_size == 0) {
    ImGui::Text("Waiting for optimization data...");
    ImGui::End();
    return;
  }

  ImGui::Text("Iterations: %zu", history_size);

  // Get cost convergence data
  std::vector<double> iterations;
  std::vector<double> costs;
  std::vector<double> cost_reductions;
  g_monitoring_state->getConvergenceData(iterations, costs, cost_reductions);

  // Get available vertical space
  float available_height = ImGui::GetContentRegionAvail().y;

  // Get factor types with colors
  auto factor_types = g_monitoring_state->getFactorTypes();
  std::map<std::string, ImVec4> color_map = {
      {"Visual (Reprojection)", ImVec4(0.2f, 0.8f, 0.2f, 1.0f)},
      {"Inertial (IMU)", ImVec4(0.8f, 0.2f, 0.2f, 1.0f)},
      {"IMU Calib (Random Walk)", ImVec4(0.8f, 0.8f, 0.2f, 1.0f)},
      {"Camera Intrinsics (Random Walk)", ImVec4(0.8f, 0.6f, 0.2f, 1.0f)},
      {"Extrinsics (Random Walk)", ImVec4(0.6f, 0.8f, 0.2f, 1.0f)},
      {"Omega Prior", ImVec4(0.2f, 0.8f, 0.8f, 1.0f)},
  };

  // Use subplots with aligned x-axes
  int num_plots = 0;
  if (!costs.empty() && (costs.front() != 0.0 || costs.back() != 0.0))
    num_plots++;
  if (!factor_types.empty())
    num_plots++;

  if (num_plots > 0) {
    // Calculate height for each subplot
    float plot_height = available_height / num_plots;

    if (ImPlot::BeginSubplots(
            "##ConvergenceSubplots",
            num_plots,
            1,
            ImVec2(-1, -1),
            ImPlotSubplotFlags_LinkCols | ImPlotSubplotFlags_LinkRows |
                ImPlotSubplotFlags_ShareItems)) {
      // Total Cost plot (if we have cost data)
      if (!costs.empty() && (costs.front() != 0.0 || costs.back() != 0.0)) {
        if (ImPlot::BeginPlot("Total Cost")) {
          ImPlot::SetupAxes("Iteration", "Cost");
          ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
          ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);
          ImPlot::PlotLine("Total Cost", iterations.data(), costs.data(), iterations.size());
          ImPlot::EndPlot();
        }
      }

      // Cost breakdown by factor type
      if (!factor_types.empty()) {
        if (ImPlot::BeginPlot("Cost Breakdown by Factor Type")) {
          ImPlot::SetupAxes("Iteration", "Cost (Sum of Squared Residuals)");
          ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
          ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);

          for (const auto& factor_type : factor_types) {
            std::vector<double> iters, costs_by_factor;
            g_monitoring_state->getCostByType(factor_type, iters, costs_by_factor);
            if (!costs_by_factor.empty()) {
              // Set color for this factor type
              auto color_it = color_map.find(factor_type);
              if (color_it != color_map.end()) {
                ImPlot::SetNextLineStyle(color_it->second);
              }
              ImPlot::PlotLine(
                  factor_type.c_str(), iters.data(), costs_by_factor.data(), iters.size());
            }
          }
          ImPlot::EndPlot();
        }
      }

      ImPlot::EndSubplots();
    }
  }

  ImGui::End();
}

void renderCalibrationTimeSeries() {
  ImGui::Begin("Calibration Parameters", nullptr, ImGuiWindowFlags_NoCollapse);

  size_t history_size = g_monitoring_state->getHistorySize();
  if (history_size == 0) {
    ImGui::Text("Waiting for calibration data...");
    ImGui::Text("(IMU params, time offset, camera distances, focal length)");
    ImGui::End();
    return;
  }

  // Get current iteration data (respects iteration selection slider)
  auto current = getCurrentIterationData();

  if (current.rig_calibrations.empty()) {
    ImGui::Text("No calibration data available for iteration %d", current.iteration);
    ImGui::End();
    return;
  }

  ImGui::Text("Iteration %d: Temporal variation of calibration parameters", current.iteration);
  ImGui::Text(
      "%zu rigs over %.2f seconds",
      current.rig_calibrations.size(),
      current.rig_calibrations.back().timestamp_sec -
          current.rig_calibrations.front().timestamp_sec);
  ImGui::Spacing();

  // Extract timestamps
  std::vector<double> timestamps;
  timestamps.reserve(current.rig_calibrations.size());
  for (const auto& rig_calib : current.rig_calibrations) {
    timestamps.push_back(rig_calib.timestamp_sec);
  }

  // Collect unique sensor indices
  std::set<int> imu_indices, camera_indices;
  std::set<std::pair<int, int>> baseline_pairs;
  for (const auto& rig_calib : current.rig_calibrations) {
    for (const auto& [idx, _] : rig_calib.imu_accel_bias)
      imu_indices.insert(idx);
    for (const auto& [idx, _] : rig_calib.camera_focal_length_x)
      camera_indices.insert(idx);
    for (const auto& [pair, _] : rig_calib.camera_baseline_distance)
      baseline_pairs.insert(pair);
  }

  // IMU Accelerometer Bias (X, Y, Z components)
  if (!imu_indices.empty()) {
    if (ImPlot::BeginPlot("IMU Accelerometer Bias", ImVec2(-1, 250))) {
      ImPlot::SetupAxes("Time (sec)", "Bias (m/s^2)");
      ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);

      for (int imu_idx : imu_indices) {
        std::vector<double> bias_x, bias_y, bias_z;
        bias_x.reserve(current.rig_calibrations.size());
        bias_y.reserve(current.rig_calibrations.size());
        bias_z.reserve(current.rig_calibrations.size());

        for (const auto& rig_calib : current.rig_calibrations) {
          auto it = rig_calib.imu_accel_bias.find(imu_idx);
          if (it != rig_calib.imu_accel_bias.end()) {
            bias_x.push_back(it->second[0]);
            bias_y.push_back(it->second[1]);
            bias_z.push_back(it->second[2]);
          }
        }

        if (!bias_x.empty()) {
          ImPlot::PlotLine(
              fmt::format("IMU{} X", imu_idx).c_str(),
              timestamps.data(),
              bias_x.data(),
              bias_x.size());
          ImPlot::PlotLine(
              fmt::format("IMU{} Y", imu_idx).c_str(),
              timestamps.data(),
              bias_y.data(),
              bias_y.size());
          ImPlot::PlotLine(
              fmt::format("IMU{} Z", imu_idx).c_str(),
              timestamps.data(),
              bias_z.data(),
              bias_z.size());
        }
      }

      ImPlot::EndPlot();
    }
  }

  // IMU Gyroscope Bias (X, Y, Z components)
  if (!imu_indices.empty()) {
    if (ImPlot::BeginPlot("IMU Gyroscope Bias", ImVec2(-1, 250))) {
      ImPlot::SetupAxes("Time (sec)", "Bias (rad/s)");
      ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);

      for (int imu_idx : imu_indices) {
        std::vector<double> bias_x, bias_y, bias_z;
        bias_x.reserve(current.rig_calibrations.size());
        bias_y.reserve(current.rig_calibrations.size());
        bias_z.reserve(current.rig_calibrations.size());

        for (const auto& rig_calib : current.rig_calibrations) {
          auto it = rig_calib.imu_gyro_bias.find(imu_idx);
          if (it != rig_calib.imu_gyro_bias.end()) {
            bias_x.push_back(it->second[0]);
            bias_y.push_back(it->second[1]);
            bias_z.push_back(it->second[2]);
          }
        }

        if (!bias_x.empty()) {
          ImPlot::PlotLine(
              fmt::format("IMU{} X", imu_idx).c_str(),
              timestamps.data(),
              bias_x.data(),
              bias_x.size());
          ImPlot::PlotLine(
              fmt::format("IMU{} Y", imu_idx).c_str(),
              timestamps.data(),
              bias_y.data(),
              bias_y.size());
          ImPlot::PlotLine(
              fmt::format("IMU{} Z", imu_idx).c_str(),
              timestamps.data(),
              bias_z.data(),
              bias_z.size());
        }
      }

      ImPlot::EndPlot();
    }
  }

  // IMU Time Offsets
  if (!imu_indices.empty()) {
    if (ImPlot::BeginPlot("IMU Time Offsets", ImVec2(-1, 250))) {
      ImPlot::SetupAxes("Time (sec)", "Time Offset (sec)");
      ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);

      for (int imu_idx : imu_indices) {
        std::vector<double> offset_accel, offset_gyro;
        offset_accel.reserve(current.rig_calibrations.size());
        offset_gyro.reserve(current.rig_calibrations.size());

        for (const auto& rig_calib : current.rig_calibrations) {
          auto it_accel = rig_calib.imu_time_offset_accel.find(imu_idx);
          auto it_gyro = rig_calib.imu_time_offset_gyro.find(imu_idx);
          if (it_accel != rig_calib.imu_time_offset_accel.end()) {
            offset_accel.push_back(it_accel->second);
          }
          if (it_gyro != rig_calib.imu_time_offset_gyro.end()) {
            offset_gyro.push_back(it_gyro->second);
          }
        }

        if (!offset_accel.empty()) {
          ImPlot::PlotLine(
              fmt::format("IMU{} Accel", imu_idx).c_str(),
              timestamps.data(),
              offset_accel.data(),
              offset_accel.size());
        }
        if (!offset_gyro.empty()) {
          ImPlot::PlotLine(
              fmt::format("IMU{} Gyro", imu_idx).c_str(),
              timestamps.data(),
              offset_gyro.data(),
              offset_gyro.size());
        }
      }

      ImPlot::EndPlot();
    }
  }

  // Camera Focal Lengths
  if (!camera_indices.empty()) {
    if (ImPlot::BeginPlot("Camera Focal Lengths", ImVec2(-1, 250))) {
      ImPlot::SetupAxes("Time (sec)", "Focal Length (pixels)");
      ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);

      for (int cam_idx : camera_indices) {
        std::vector<double> focal_x, focal_y;
        focal_x.reserve(current.rig_calibrations.size());
        focal_y.reserve(current.rig_calibrations.size());

        for (const auto& rig_calib : current.rig_calibrations) {
          auto it_x = rig_calib.camera_focal_length_x.find(cam_idx);
          auto it_y = rig_calib.camera_focal_length_y.find(cam_idx);
          if (it_x != rig_calib.camera_focal_length_x.end()) {
            focal_x.push_back(it_x->second);
          }
          if (it_y != rig_calib.camera_focal_length_y.end()) {
            focal_y.push_back(it_y->second);
          }
        }

        if (!focal_x.empty()) {
          ImPlot::PlotLine(
              fmt::format("Cam{} fx", cam_idx).c_str(),
              timestamps.data(),
              focal_x.data(),
              focal_x.size());
        }
        if (!focal_y.empty()) {
          ImPlot::PlotLine(
              fmt::format("Cam{} fy", cam_idx).c_str(),
              timestamps.data(),
              focal_y.data(),
              focal_y.size());
        }
      }

      ImPlot::EndPlot();
    }
  }

  // Camera Time Offsets
  if (!camera_indices.empty()) {
    if (ImPlot::BeginPlot("Camera Time Offsets", ImVec2(-1, 250))) {
      ImPlot::SetupAxes("Time (sec)", "Time Offset (sec)");
      ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);

      for (int cam_idx : camera_indices) {
        std::vector<double> time_offset;
        time_offset.reserve(current.rig_calibrations.size());

        for (const auto& rig_calib : current.rig_calibrations) {
          auto it = rig_calib.camera_time_offset.find(cam_idx);
          if (it != rig_calib.camera_time_offset.end()) {
            time_offset.push_back(it->second);
          }
        }

        if (!time_offset.empty()) {
          ImPlot::PlotLine(
              fmt::format("Cam{}", cam_idx).c_str(),
              timestamps.data(),
              time_offset.data(),
              time_offset.size());
        }
      }

      ImPlot::EndPlot();
    }
  }

  // Camera-to-Camera Baseline Distances
  if (!baseline_pairs.empty()) {
    if (ImPlot::BeginPlot("Camera-to-Camera Baseline Distances", ImVec2(-1, 250))) {
      ImPlot::SetupAxes("Time (sec)", "Distance (m)");
      ImPlot::SetupLegend(ImPlotLocation_NorthEast, ImPlotLegendFlags_Outside);

      for (const auto& [cam_i, cam_j] : baseline_pairs) {
        std::vector<double> distances;
        distances.reserve(current.rig_calibrations.size());

        for (const auto& rig_calib : current.rig_calibrations) {
          auto it = rig_calib.camera_baseline_distance.find({cam_i, cam_j});
          if (it != rig_calib.camera_baseline_distance.end()) {
            distances.push_back(it->second);
          }
        }

        if (!distances.empty()) {
          ImPlot::PlotLine(
              fmt::format("Cam{} to Cam{}", cam_i, cam_j).c_str(),
              timestamps.data(),
              distances.data(),
              distances.size());
        }
      }

      ImPlot::EndPlot();
    }
  }

  ImGui::End();
}

void render3DTrajectory() {
  auto current = getCurrentIterationData();

  if (current.camera_positions.empty()) {
    ImGui::Begin("3D Trajectory & Points", nullptr, ImGuiWindowFlags_NoCollapse);
    ImGui::Text("Waiting for trajectory data...");
    ImGui::End();
    return;
  }

  // Convert to float vectors for geometry buffers
  std::vector<Eigen::Vector3f> trajectory_positions;
  trajectory_positions.reserve(current.camera_positions.size());
  for (const auto& pos : current.camera_positions) {
    trajectory_positions.push_back(pos.cast<float>());
  }

  std::vector<Eigen::Vector3f> point_positions;
  point_positions.reserve(current.point_positions.size());
  for (const auto& pos : current.point_positions) {
    point_positions.push_back(pos.cast<float>());
  }

  // Update geometry buffers
  geometry::updateBuffer(g_trajectory_buf, trajectory_positions);
  geometry::updateBuffer(g_points_buf, point_positions);

  // Auto-fit camera only on first update
  static bool camera_initialized = false;
  if (!camera_initialized && !trajectory_positions.empty()) {
    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    for (const auto& v : trajectory_positions) {
      center += v;
    }
    center /= trajectory_positions.size();

    // Calculate bounding radius
    float max_dist = 0.0f;
    for (const auto& v : trajectory_positions) {
      max_dist = std::max(max_dist, (v - center).norm());
    }

    // Position camera to view the trajectory
    float cam_distance = std::max(max_dist * 2.5f, 1.0f);
    g_3d_viewer->camera().position =
        center + Eigen::Vector3f(-cam_distance, -cam_distance * 0.5f, cam_distance * 0.4f);

    // Make camera look toward the center
    Eigen::Vector3f look_direction = (center - g_3d_viewer->camera().position).normalized();
    float horizontal_dist = std::sqrt(
        look_direction.x() * look_direction.x() + look_direction.y() * look_direction.y());
    g_3d_viewer->camera().azimuth = std::atan2(look_direction.y(), look_direction.x());
    g_3d_viewer->camera().elevation = std::atan2(look_direction.z(), horizontal_dist);

    camera_initialized = true;
  }

  // Render using immediate-mode API
  auto drawCb = [](const Eigen::Matrix4f& mvp) {
    geometry::drawLineStrip(g_trajectory_buf, mvp, Eigen::Vector3f(1.0f, 0.5f, 0.0f));
    geometry::drawPoints(g_points_buf, mvp, Eigen::Vector3f(0.3f, 0.7f, 0.3f), 0.6f);
  };
  g_3d_viewer->render("3D Trajectory & Points", drawCb);
}

void renderResidualHistograms() {
  ImGui::Begin("Residual Histograms", nullptr, ImGuiWindowFlags_NoCollapse);

  auto current = getCurrentIterationData();

  if (current.residuals_by_type.empty()) {
    ImGui::Text("Waiting for residual data...");
    ImGui::End();
    return;
  }

  // Cache global max P95 computation (only update when iteration changes)
  static int last_history_size = -1;
  static std::map<std::string, double> cached_global_max_p95_by_type;

  size_t current_history_size = g_monitoring_state->getHistorySize();
  if (last_history_size != static_cast<int>(current_history_size)) {
    cached_global_max_p95_by_type = g_monitoring_state->getGlobalMaxP95ByType();
    last_history_size = current_history_size;
  }

  // Define explicit ordering: Visual and IMU first, then others
  std::vector<std::string> factor_order = {
      "Visual (Reprojection)",
      "Inertial (IMU)",
      "IMU Calib (Random Walk)",
      "Camera Intrinsics (Random Walk)",
      "Extrinsics (Random Walk)",
      "Omega Prior"};

  // Color mapping for different factor types
  std::map<std::string, ImVec4> color_map = {
      {"Visual (Reprojection)", ImVec4(0.2f, 0.8f, 0.2f, 1.0f)}, // Green
      {"Inertial (IMU)", ImVec4(0.8f, 0.2f, 0.2f, 1.0f)}, // Red
      {"IMU Calib (Random Walk)", ImVec4(0.8f, 0.8f, 0.2f, 1.0f)}, // Yellow
      {"Camera Intrinsics (Random Walk)", ImVec4(0.8f, 0.6f, 0.2f, 1.0f)}, // Orange
      {"Extrinsics (Random Walk)", ImVec4(0.6f, 0.8f, 0.2f, 1.0f)}, // Yellow-green
      {"Omega Prior", ImVec4(0.2f, 0.8f, 0.8f, 1.0f)}, // Cyan
  };

  // Count how many factor types actually exist in current data
  int num_factors = 0;
  for (const auto& factor_type : factor_order) {
    if (current.residuals_by_type.find(factor_type) != current.residuals_by_type.end()) {
      num_factors++;
    }
  }

  // Calculate dynamic dimensions to fill available space
  const int num_columns = 2;
  const int num_rows = (num_factors + num_columns - 1) / num_columns; // Ceiling division
  float available_width = ImGui::GetContentRegionAvail().x;
  float available_height = ImGui::GetContentRegionAvail().y;

  float plot_width = (available_width / num_columns) - 10.0f;
  float plot_height = num_rows > 0 ? (available_height / num_rows) - 80.0f
                                   : 300.0f; // Reserve space for text labels

  int column = 0;

  // Iterate in defined order
  for (const auto& factor_type : factor_order) {
    // Check if this factor type exists in current data
    auto residuals_it = current.residuals_by_type.find(factor_type);
    if (residuals_it == current.residuals_by_type.end()) {
      continue; // Skip if not present
    }

    const auto& residuals = residuals_it->second;

    if (column > 0) {
      ImGui::SameLine();
    }

    ImGui::BeginGroup();

    // Get color for this factor type
    ImVec4 color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f); // Default gray
    auto color_it = color_map.find(factor_type);
    if (color_it != color_map.end()) {
      color = color_it->second;
    }

    // Show factor type name with color
    ImGui::TextColored(color, "%s", factor_type.c_str());
    ImGui::Text("Count: %zu", residuals.size());

    // Use precomputed statistics from IterationData (avoid recomputation)
    if (!residuals.empty()) {
      // Get precomputed mean from IterationData
      double mean = 0.0;
      auto mean_it = current.residuals_mean_by_type.find(factor_type);
      if (mean_it != current.residuals_mean_by_type.end()) {
        mean = mean_it->second;
      }

      // Get precomputed p95 from IterationData
      double p95 = 0.0;
      auto p95_it = current.residuals_p95_by_type.find(factor_type);
      if (p95_it != current.residuals_p95_by_type.end()) {
        p95 = p95_it->second;
      }

      // Still compute min/max for display (fast single pass)
      double max_val = *std::max_element(residuals.begin(), residuals.end());
      double min_val = *std::min_element(residuals.begin(), residuals.end());

      ImGui::Text("Mean: %.3f  P95: %.3f  Max: %.3f", mean, p95, max_val);

      // Get global max P95 for this factor type (from cache)
      double global_max_p95 = 1.0; // default
      auto global_max_p95_it = cached_global_max_p95_by_type.find(factor_type);
      if (global_max_p95_it != cached_global_max_p95_by_type.end()) {
        global_max_p95 = global_max_p95_it->second;
      }

      // Downsample large residual arrays for faster rendering
      const size_t max_histogram_samples = 10000;
      std::vector<double> sampled_residuals;
      const double* data_ptr;
      size_t data_size;

      if (residuals.size() > max_histogram_samples) {
        // Downsample by taking every Nth element
        size_t stride = residuals.size() / max_histogram_samples;
        sampled_residuals.reserve(max_histogram_samples);
        for (size_t i = 0; i < residuals.size(); i += stride) {
          sampled_residuals.push_back(residuals[i]);
        }
        data_ptr = sampled_residuals.data();
        data_size = sampled_residuals.size();
      } else {
        data_ptr = residuals.data();
        data_size = residuals.size();
      }

      // Plot histogram using ImPlot with density normalization and dynamic sizing
      if (ImPlot::BeginPlot(("##hist_" + factor_type).c_str(), ImVec2(plot_width, plot_height))) {
        ImPlot::SetupAxes(
            "Residual Error", "Density", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupLegend(ImPlotLocation_North, ImPlotLegendFlags_None); // Hide legend
        ImPlot::PushStyleColor(ImPlotCol_Fill, color);
        ImPlot::PlotHistogram(
            factor_type.c_str(),
            data_ptr,
            data_size,
            ImPlotBin_Scott, // bins
            1.0, // density normalization (true)
            ImPlotRange(0, std::max(global_max_p95, 0.0))); // range from global max P95
        ImPlot::PopStyleColor();

        ImPlot::EndPlot();
      }
    }

    ImGui::EndGroup();

    column++;
    if (column >= num_columns) {
      column = 0;
    }
  }

  ImGui::End();
}

void frame() {
  const int width = sapp_width();
  const int height = sapp_height();

  // Start new ImGui frame
  simgui_new_frame({width, height, sapp_frame_duration(), sapp_dpi_scale()});

  // Render menu bar
  renderMenuBar();

  // Create fullscreen dockspace
  ImGuiViewport* viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::SetNextWindowViewport(viewport->ID);

  ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking;
  window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse;
  window_flags |= ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
  window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
  window_flags |= ImGuiWindowFlags_NoBackground;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

  ImGui::Begin("DockSpace", nullptr, window_flags);
  ImGui::PopStyleVar(3);

  ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");

  // Build the docking layout on first run
  static bool first_time = true;
  if (first_time) {
    first_time = false;

    ImGui::DockBuilderRemoveNode(dockspace_id); // Clear any previous layout
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, viewport->WorkSize);

    // Split the dockspace
    ImGuiID dock_left, dock_right;
    ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.2f, &dock_left, &dock_right);

    ImGuiID dock_right_top, dock_right_bottom;
    ImGui::DockBuilderSplitNode(
        dock_right, ImGuiDir_Up, 0.35f, &dock_right_top, &dock_right_bottom);

    ImGuiID dock_deepdive_left, dock_deepdive_right;
    ImGui::DockBuilderSplitNode(
        dock_right_bottom, ImGuiDir_Left, 0.5f, &dock_deepdive_left, &dock_deepdive_right);

    // Dock windows to their positions
    ImGui::DockBuilderDockWindow("Controls & Status", dock_left);
    ImGui::DockBuilderDockWindow("Convergence", dock_right_top);
    ImGui::DockBuilderDockWindow(
        "Calibration Parameters", dock_right_top); // Tabbed with Convergence
    ImGui::DockBuilderDockWindow("3D Trajectory & Points", dock_deepdive_left);
    ImGui::DockBuilderDockWindow("Residual Histograms", dock_deepdive_right);

    ImGui::DockBuilderFinish(dockspace_id);
  }

  ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
  ImGui::End();

  // Show ImGui demo for debugging (off by default)
  if (g_show_imgui_demo) {
    ImGui::ShowDemoWindow(&g_show_imgui_demo);
  }

  // Render all panels
  renderControlsAndStatusPanel();
  renderConvergencePlots();
  renderCalibrationTimeSeries();
  render3DTrajectory();
  renderResidualHistograms();

  // render sokol_gfx debugger
  sgimgui_draw();

  // Get ImGui draw data and print vertex/index counts before rendering
  ImGui::Render();

  // Begin rendering
  sg_begin_pass({.action = pass_action, .swapchain = sglue_swapchain()});

  // Render ImGui
  simgui_render();

  sg_end_pass();
  sg_commit();
}

void cleanup() {
  // Wait for optimization thread to finish
  if (g_optimization_thread.joinable()) {
    g_optimization_thread.join();
  }

  // Cleanup geometry buffers
  geometry::destroyBuffer(g_trajectory_buf);
  geometry::destroyBuffer(g_points_buf);
  geometry::shutdown();

  g_3d_viewer.reset();
  ImPlot::DestroyContext();
  sgimgui_shutdown();
  simgui_shutdown();
  sg_shutdown();
}

void event(const sapp_event* ev) {
  simgui_handle_event(ev);
}

sapp_desc sokol_main(int argc, char* argv[]) {
  // Initialize monitoring state FIRST (before any threads)
  g_monitoring_state = std::make_shared<MonitoringState>();

  // Parse command line arguments for optimization
  std::filesystem::path filterPathIn;
  std::filesystem::path outputsDir;
  InitSettings initSettings;
  OptimizerSettings optSettings;
  int64_t rigIndexStart = -1, rigIndexEnd = -1;

  CLI::App app{"Visual Inertial Bundle Adjustment GUI Monitor"};
  initSettings.addCommandLine(app);
  optSettings.addCommandLine(app);
  app.add_option("-i,--input-dir", filterPathIn, "input folder path");
  app.add_option("-o,--output-dir", outputsDir, "Output folder path");
  app.add_option("--rig-start", rigIndexStart, "Index of starting rig in recording");
  app.add_option("--rig-end", rigIndexEnd, "Index of ending rig in recording");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    std::cerr << "Command line parse error: " << e.what() << std::endl;
  }

  // Start optimization thread if input path provided
  // g_monitoring_state is now safely initialized above
  if (!filterPathIn.empty()) {
    g_optimization_thread = std::thread(
        runOptimizationThread,
        filterPathIn,
        outputsDir,
        initSettings,
        optSettings,
        rigIndexStart,
        rigIndexEnd,
        g_monitoring_state);
  }

  sapp_desc desc = {};
  desc.init_cb = init;
  desc.frame_cb = frame;
  desc.cleanup_cb = cleanup;
  desc.event_cb = event;
  desc.window_title = "Aria ViBa Monitor";
  desc.width = 1600;
  desc.height = 1000;
  desc.logger.func = slog_func;
  desc.icon.sokol_default = true;

  return desc;
}
