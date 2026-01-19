/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-DetailCall
#include <io/SaveDeviceTrajectory.h>

#define DEFAULT_LOG_CHANNEL "ViBa::SaveDeviceTrajectory"
#include <logging/Log.h>

namespace visual_inertial_ba {

constexpr std::array<const char*, 20> kOpenLoopTrajectoryColumns = {
    "tracking_timestamp_us",
    "utc_timestamp_ns",
    "session_uid",
    "tx_odometry_device",
    "ty_odometry_device",
    "tz_odometry_device",
    "qx_odometry_device",
    "qy_odometry_device",
    "qz_odometry_device",
    "qw_odometry_device",
    "device_linear_velocity_x_odometry",
    "device_linear_velocity_y_odometry",
    "device_linear_velocity_z_odometry",
    "angular_velocity_x_device",
    "angular_velocity_y_device",
    "angular_velocity_z_device",
    "gravity_x_odometry",
    "gravity_y_odometry",
    "gravity_z_odometry",
    "quality_score",
};

void saveOpenLoopTrajectory(
    const SingleSessionProblem& prob,
    const SessionData& fData,
    const std::filesystem::path& outputPath) {
  auto rigIndices = prob.sortedRigIndices();

  std::ofstream ofs(outputPath);
  for (size_t i = 0; i < kOpenLoopTrajectoryColumns.size(); i++) {
    ofs << (i == 0 ? "" : ",") << kOpenLoopTrajectoryColumns[i];
  }
  ofs << std::endl;

  const SE3& T_bodyImu_device = fData.slamInfo.T_bodyImu_device;
  for (int64_t rigIndex : rigIndices) {
    const auto& rigVar = prob.inertialPose(rigIndex);
    const auto& inputRig = fData.inertialPoses.at(rigIndex);

    SE3 T_odometry_device = rigVar.T_bodyImu_world.value.inverse() * T_bodyImu_device;
    Vec3 t_odometry_device = T_odometry_device.translation();
    Vec4 q_odometry_device = T_odometry_device.so3().unit_quaternion().coeffs();
    Vec3 deviceLinearVelocity_odometry = //
        rigVar.vel_world.value +
        rigVar.T_bodyImu_world.value.so3().inverse() *
            rigVar.omega.value.cross(T_bodyImu_device.translation());
    Vec3 angularVelocity_device = T_bodyImu_device.so3().inverse() * rigVar.omega.value;
    Vec3 gravity_odometry = prob.gravityWorld().value.vec;

    ofs << inputRig.timestamp_us << "," // tracking_timestamp_us
        << inputRig.utc_timestamp_ns << "," // utc_timestamp_ns
        << inputRig.sessionOrGraphUid << "," // session_uid
        << t_odometry_device.x() << "," // tx_odometry_device
        << t_odometry_device.y() << "," // ty_odometry_device
        << t_odometry_device.z() << "," // tz_odometry_device
        << q_odometry_device.x() << "," // qx_odometry_device
        << q_odometry_device.y() << "," // qy_odometry_device
        << q_odometry_device.z() << "," // qz_odometry_device
        << q_odometry_device.w() << "," // qw_odometry_device
        << deviceLinearVelocity_odometry.x() << "," // device_linear_velocity_x_odometry
        << deviceLinearVelocity_odometry.y() << "," // device_linear_velocity_y_odometry
        << deviceLinearVelocity_odometry.z() << "," // device_linear_velocity_z_odometry
        << angularVelocity_device.x() << "," // angular_velocity_x_device
        << angularVelocity_device.y() << "," // angular_velocity_y_device
        << angularVelocity_device.z() << "," // angular_velocity_z_device
        << gravity_odometry.x() << "," // gravity_x_odometry
        << gravity_odometry.y() << "," // gravity_y_odometry
        << gravity_odometry.z() << "," // gravity_z_odometry
        << inputRig.qualityScore << std::endl; // quality_score
  }

  XR_LOGI(
      "Device trajectory ({} entries) saved (as open loop) to {}",
      rigIndices.size(),
      outputPath.string());
}

constexpr std::array<const char*, 20> kCloseLoopTrajectoryColumns = {
    "graph_uid",
    "tracking_timestamp_us",
    "utc_timestamp_ns",
    "tx_world_device",
    "ty_world_device",
    "tz_world_device",
    "qx_world_device",
    "qy_world_device",
    "qz_world_device",
    "qw_world_device",
    "device_linear_velocity_x_device",
    "device_linear_velocity_y_device",
    "device_linear_velocity_z_device",
    "angular_velocity_x_device",
    "angular_velocity_y_device",
    "angular_velocity_z_device",
    "gravity_x_world",
    "gravity_y_world",
    "gravity_z_world",
    "quality_score",
};

void saveCloseLoopTrajectory(
    const SingleSessionProblem& prob,
    const SessionData& fData,
    const std::filesystem::path& outputPath) {
  auto rigIndices = prob.sortedRigIndices();

  std::ofstream ofs(outputPath);
  for (size_t i = 0; i < kCloseLoopTrajectoryColumns.size(); i++) {
    ofs << (i == 0 ? "" : ",") << kCloseLoopTrajectoryColumns[i];
  }
  ofs << std::endl;

  const SE3& T_bodyImu_device = fData.slamInfo.T_bodyImu_device;
  for (int64_t rigIndex : rigIndices) {
    const auto& rigVar = prob.inertialPose(rigIndex);
    const auto& inputRig = fData.inertialPoses.at(rigIndex);

    SE3 T_world_device = rigVar.T_bodyImu_world.value.inverse() * T_bodyImu_device;
    Vec3 t_world_device = T_world_device.translation();
    Vec4 q_world_device = T_world_device.so3().unit_quaternion().coeffs();
    Vec3 deviceLinearVelocity_device = //
        T_bodyImu_device.so3().inverse() *
        (rigVar.T_bodyImu_world.value.so3() * rigVar.vel_world.value +
         rigVar.omega.value.cross(T_bodyImu_device.translation()));
    Vec3 angularVelocity_device = T_bodyImu_device.so3().inverse() * rigVar.omega.value;
    Vec3 gravity_odometry = prob.gravityWorld().value.vec;

    ofs << inputRig.sessionOrGraphUid << "," // graph_uid
        << inputRig.timestamp_us << "," // tracking_timestamp_us
        << inputRig.utc_timestamp_ns << "," // utc_timestamp_ns
        << t_world_device.x() << "," // tx_world_device
        << t_world_device.y() << "," // ty_world_device
        << t_world_device.z() << "," // tz_world_device
        << q_world_device.x() << "," // qx_world_device
        << q_world_device.y() << "," // qy_world_device
        << q_world_device.z() << "," // qz_world_device
        << q_world_device.w() << "," // qw_world_device
        << deviceLinearVelocity_device.x() << "," // device_linear_velocity_x_device
        << deviceLinearVelocity_device.y() << "," // device_linear_velocity_y_device
        << deviceLinearVelocity_device.z() << "," // device_linear_velocity_z_device
        << angularVelocity_device.x() << "," // angular_velocity_x_device
        << angularVelocity_device.y() << "," // angular_velocity_y_device
        << angularVelocity_device.z() << "," // angular_velocity_z_device
        << gravity_odometry.x() << "," // gravity_x_odometry
        << gravity_odometry.y() << "," // gravity_y_odometry
        << gravity_odometry.z() << "," // gravity_z_odometry
        << inputRig.qualityScore << std::endl; // quality_score
  }

  XR_LOGI(
      "Device trajectory ({} entries) saved (as close loop) to {}",
      rigIndices.size(),
      outputPath.string());
}

} // namespace visual_inertial_ba
