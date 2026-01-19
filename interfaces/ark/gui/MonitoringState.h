/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace viba_gui {

struct IterationData {
  int iteration = 0;
  double cost = 0.0;
  double cost_reduction = 0.0;
  double gradient_norm = 0.0;
  double lambda = 0.0;
  bool step_accepted = false;
  std::chrono::milliseconds time_ms{0};

  // Raw residual data: factor type -> vector of residual errors
  std::map<std::string, std::vector<double>> residuals_by_type;

  // Precomputed statistics for histogram display and convergence tracking
  std::map<std::string, double> residuals_p95_by_type; // 95th percentile
  std::map<std::string, double> residuals_mean_by_type; // mean error
  std::map<std::string, int64_t> residuals_count_by_type; // count
  std::map<std::string, double>
      cost_by_type; // total cost (sum of squared residuals) per factor type

  // Trajectory data
  std::vector<Eigen::Vector3d> camera_positions;
  std::vector<Eigen::Quaterniond> camera_orientations;

  // 3D point cloud data
  std::vector<Eigen::Vector3d> point_positions;

  // Calibration parameters per rig (temporal variation)
  // Maps rig_index -> (sensor_index -> value)
  struct RigCalibration {
    int64_t rig_index = 0;
    double timestamp_sec = 0.0; // Timestamp for this rig
    std::map<int, Eigen::Vector3d> imu_accel_bias; // IMU index -> accel bias (m/s^2)
    std::map<int, Eigen::Vector3d> imu_gyro_bias; // IMU index -> gyro bias (rad/s)
    std::map<int, double> imu_time_offset_accel; // IMU index -> accel time offset (sec)
    std::map<int, double> imu_time_offset_gyro; // IMU index -> gyro time offset (sec)
    std::map<int, double> camera_focal_length_x; // Camera index -> fx
    std::map<int, double> camera_focal_length_y; // Camera index -> fy
    std::map<int, double> camera_time_offset; // Camera index -> time offset (sec)
    std::map<std::pair<int, int>, double> camera_baseline_distance; // (cam1, cam2) -> distance (m)
  };
  std::vector<RigCalibration> rig_calibrations; // Calibrations for each rig (sorted by rig_index)
};

struct ProblemStats {
  int64_t num_rigs = 0;
  int64_t num_point_tracks = 0;
  int64_t num_cameras = 0;
  int64_t num_extrinsics = 0;
  int64_t num_imu_calibs = 0;
  int64_t recording_length_us = 0;
};

class MonitoringState {
 public:
  void pushIteration(const IterationData& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    history_.push_back(data);
  }

  void setProblemStats(const ProblemStats& stats) {
    std::lock_guard<std::mutex> lock(mutex_);
    problem_stats_ = stats;
  }

  void setOptimizationRunning(bool running) {
    std::lock_guard<std::mutex> lock(mutex_);
    optimization_running_ = running;
  }

  void setOptimizationDone(bool done) {
    std::lock_guard<std::mutex> lock(mutex_);
    optimization_done_ = done;
  }

  std::vector<IterationData> getHistory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return history_;
  }

  size_t getHistorySize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return history_.size();
  }

  IterationData getIteration(size_t index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (index >= history_.size()) {
      return IterationData{};
    }
    return history_[index];
  }

  // Get convergence data without copying heavy point/residual data
  void getConvergenceData(
      std::vector<double>& iterations,
      std::vector<double>& costs,
      std::vector<double>& cost_reductions) const {
    std::lock_guard<std::mutex> lock(mutex_);
    iterations.clear();
    costs.clear();
    cost_reductions.clear();
    iterations.reserve(history_.size());
    costs.reserve(history_.size());
    cost_reductions.reserve(history_.size());

    for (const auto& iter : history_) {
      iterations.push_back(static_cast<double>(iter.iteration));
      costs.push_back(iter.cost);
      cost_reductions.push_back(iter.cost_reduction);
    }
  }

  // Get residual statistics across iterations for a specific factor type
  void getResidualStats(
      const std::string& factor_type,
      std::vector<double>& iterations,
      std::vector<double>& means,
      std::vector<int64_t>& counts) const {
    std::lock_guard<std::mutex> lock(mutex_);
    iterations.clear();
    means.clear();
    counts.clear();
    iterations.reserve(history_.size());
    means.reserve(history_.size());
    counts.reserve(history_.size());

    for (const auto& iter : history_) {
      auto mean_it = iter.residuals_mean_by_type.find(factor_type);
      auto count_it = iter.residuals_count_by_type.find(factor_type);
      if (mean_it != iter.residuals_mean_by_type.end() &&
          count_it != iter.residuals_count_by_type.end()) {
        iterations.push_back(static_cast<double>(iter.iteration));
        means.push_back(mean_it->second);
        counts.push_back(count_it->second);
      }
    }
  }

  // Get cost breakdown by factor type across iterations
  void getCostByType(
      const std::string& factor_type,
      std::vector<double>& iterations,
      std::vector<double>& costs) const {
    std::lock_guard<std::mutex> lock(mutex_);
    iterations.clear();
    costs.clear();
    iterations.reserve(history_.size());
    costs.reserve(history_.size());

    for (const auto& iter : history_) {
      auto cost_it = iter.cost_by_type.find(factor_type);
      if (cost_it != iter.cost_by_type.end()) {
        iterations.push_back(static_cast<double>(iter.iteration));
        costs.push_back(cost_it->second);
      }
    }
  }

  // Get all factor types that have been recorded
  std::set<std::string> getFactorTypes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::set<std::string> types;
    if (!history_.empty()) {
      for (const auto& [factor_type, _] : history_.back().residuals_by_type) {
        types.insert(factor_type);
      }
    }
    return types;
  }

  // Get global max residual value for each factor type across all iterations
  std::map<std::string, double> getGlobalMaxByType() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<std::string, double> global_max;

    for (const auto& iter : history_) {
      for (const auto& [factor_type, residuals] : iter.residuals_by_type) {
        if (!residuals.empty()) {
          double max_val = *std::max_element(residuals.begin(), residuals.end());
          auto it = global_max.find(factor_type);
          if (it == global_max.end()) {
            global_max[factor_type] = max_val;
          } else {
            global_max[factor_type] = std::max(global_max[factor_type], max_val);
          }
        }
      }
    }

    return global_max;
  }

  // Get global max P95 residual value for each factor type across all iterations
  std::map<std::string, double> getGlobalMaxP95ByType() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<std::string, double> global_max_p95;

    for (const auto& iter : history_) {
      for (const auto& [factor_type, p95_val] : iter.residuals_p95_by_type) {
        auto it = global_max_p95.find(factor_type);
        if (it == global_max_p95.end()) {
          global_max_p95[factor_type] = p95_val;
        } else {
          global_max_p95[factor_type] = std::max(global_max_p95[factor_type], p95_val);
        }
      }
    }

    return global_max_p95;
  }

  IterationData getLatest() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (history_.empty()) {
      return IterationData{};
    }
    return history_.back();
  }

  ProblemStats getProblemStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return problem_stats_;
  }

  bool isOptimizationRunning() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return optimization_running_;
  }

  bool isOptimizationDone() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return optimization_done_;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    history_.clear();
    optimization_running_ = false;
    optimization_done_ = false;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<IterationData> history_;
  ProblemStats problem_stats_;
  bool optimization_running_ = false;
  bool optimization_done_ = false;
};

} // namespace viba_gui
