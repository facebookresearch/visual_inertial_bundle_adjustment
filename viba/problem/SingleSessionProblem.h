/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <camera_model/CameraModelParam.h>
#include <imu_model/ImuCalibParam.h>
#include <preint/PreIntegration.h>
#include <rolling_shutters/RollingShutterData.h>
#include <small_thing/Optimizer.h>
#include <viba/common/Constants.h>
#include <viba/problem/Types.h>
#include <optional>

namespace visual_inertial_ba {

using small_thing::isNull;
using small_thing::NullRef;

using pair_hash = small_thing::pair_hash;

struct InertialPoseVariables {
  small_thing::Variable<SE3> T_bodyImu_world;
  small_thing::Variable<Vec3> vel_world;
  small_thing::Variable<Vec3> omega; // angular velocity (in bodyImu's reference frame)
  int64_t timestampUs = -1;
};

// used for window of rigs sharing the same calibration params
template <typename D>
struct GroupedCalibrationVariable {
  small_thing::Variable<D> var;
  int64_t averageTimestampUs = -1;
  int64_t prevVarIndex = -1;
  int sensorIndex = -1;
};

using PointVariable = small_thing::Variable<Vec3>;
using Point2DVariable = small_thing::Variable<Vec2>;
using CameraModelVariable = GroupedCalibrationVariable<CameraModelParam>;
using ExtrinsicsVariable = GroupedCalibrationVariable<SE3>;
using ImuCalibrationVariable = GroupedCalibrationVariable<ImuCalibParam>;
using GravityData = small_thing::S2;
using GravityVariable = small_thing::Variable<GravityData>;

using SoftLossType = small_thing::HuberLossWithCutoff;

enum Verbosity { Muted, Normal, Verbose };

// class storing the optimization problem
class SingleSessionProblem {
 public:
  using RigSensorToIndex = std::unordered_map<RigSensorIndices, int64_t, pair_hash>;

  explicit SingleSessionProblem(small_thing::Optimizer& opt, Verbosity verbosity = Normal);

  // set all rigs to constant
  void setAllRigsConstant(bool constPose = true, bool constLinVel = true, bool constAngVel = true);

  // add visual factor
  void addVisualFactor(
      int64_t rigIndex,
      int64_t cameraIndex,
      PointVariable& pointTrackVar,
      const Vec2& projBaseRes,
      const Mat22& sqrtH_BaseRes,
      const SoftLossType& reprojErrorLoss);

  // add visual factor, with image detector bias
  void addVisualFactorWithBias(
      int64_t rigIndex,
      int64_t cameraIndex,
      PointVariable& pointTrackVar,
      Point2DVariable& detectorBias,
      const Vec2& projBaseRes,
      const Mat22& sqrtH_BaseRes,
      const SoftLossType& reprojErrorLoss);

  // util, print variables describing detector biases
  static void printDetectorBiases(
      const std::string& label,
      const std::vector<Point2DVariable>& vars);

  // print detector biases for observations in this tracking problem
  void printDetectorBiases(const std::string& label);

  // adds an inertial factor
  void addInertialFactor(
      int64_t prevRigIndex,
      int64_t nextRigIndex,
      int imuIndex,
      const PreIntegrationData& preintegrationData,
      const SoftLossType& imuErrorLoss);

  // add a prior on omega (angular velocity)
  void addOmegaPriorFactor(int64_t rigIndex, int imuIndex, const Vec3& omegaRadSec_Imu);

  // random walk, imu calib
  void addImuCalibRWFactor(int64_t calibPrevIndex, int64_t calibNextIndex, const VecX& diagSqrtH);

  // random walk, camera intrinsics
  void addCamIntrRWFactor(int64_t calibPrevIndex, int64_t calibNextIndex, const VecX& diagSqrtH);

  // random walk, imu extrinsics
  void addImuExtrRWFactor(int64_t calibPrevIndex, int64_t calibNextIndex, const Vec6& diagSqrtH);

  // random walk, camera extrinsics
  void addCamExtrRWFactor(int64_t calibPrevIndex, int64_t calibNextIndex, const Vec6& diagSqrtH);

  // set 6DoF prior on pose
  void addPosePrior(int64_t rigIndex, const SE3& prior_T_bodyImu_world, const Mat66& H);

  // prior on imu calib
  void
  addImuPrior(int64_t imuCalibIndex, const ImuMeasurementModelParameters& prior, const VecX& diagH);

  // prior on camera intrinsics
  void
  addCamIntrinsicsPrior(int64_t camModelIndex, const CameraModelParam& prior, const VecX& diagH);

  // prior on imu extrinsics
  void addImuExtrinsicsPrior(int64_t camExtrinsicsIndex, const SE3& prior, const Vec6& diagH);

  // prior on camera extrinsics
  void addCamExtrinsicsPrior(int64_t camExtrinsicsIndex, const SE3& prior, const Vec6& diagH);

  // clear all pose priors, returns the number of factors removed
  int64_t clearPosePriors();

  // add a prior identical to currently estimated value
  void constrainPositionAndYaw(int64_t rigIndex);

  // show histogram of all error terms
  void showHistogram(bool simpleStats) const;

  // register points (for Schur complement trick)
  void registerPointVariables();

  // compute covariance of all inertial poses (9DoF) and calibration parameters
  void computeCovariances();

  // number of rigs
  int64_t numRigVariables() const;

  // number of camera model variables
  int64_t numCameraModelVariables() const;

  // number of extrinsics (imu and camera)
  int64_t numExtrinsicsVariables() const;

  // number of camera extrinsics
  int64_t numCameraExtrinsicsVariables() const;

  // number of imu extrinsics
  int64_t numImuExtrinsicsVariables() const;

  // number of imu calibration
  int64_t numImuCalibVariables() const;

  // number of point tracks
  int64_t numPointTrackVariables() const;

  // utils to possibly recover what tracking problem owns a cost term
  // bool isOwnRigVar(small_thing::VarBase* var); // unimplemented, not needed ATM

  bool isOwnPointVar(small_thing::VarBase* var) const;

  bool isOwnImuCalibVar(small_thing::VarBase* var) const;

  bool isOwnImuExtrinsicsVar(small_thing::VarBase* var) const;

  bool isOwnCamExtrinsicsVar(small_thing::VarBase* var) const;

  bool isOwnCameraModelVar(small_thing::VarBase* var) const;

  // apply transformation to whole world
  void applyWorldTransformation(const SE3& T_newW_world);

  // return index of first rig
  std::optional<int64_t> firstRigIndex() const;

  // return sorted rig indices
  std::vector<int64_t> sortedRigIndices() const;

  // get map {index -> variable covariance}
  const std::unordered_map<int64_t, Eigen::MatrixXd>& variableCovariances() const;

  // get reference to poses
  const std::unordered_map<int64_t, InertialPoseVariables>& inertialPoses() const;

  // check if pose exists
  bool inertialPose_exists(int64_t rigIndex);

  // set pose values, create if doesn't exist
  void inertialPose_set(
      int64_t rigIndex,
      const SE3& T_bodyImu_world,
      const Vec3& vel_world,
      const Vec3& omega,
      int64_t timestampUs);

  // get inertial pose
  InertialPoseVariables& inertialPose(int64_t rigIndex);

  // get inertial pose (overload)
  const InertialPoseVariables& inertialPose(int64_t rigIndex) const;

  // get inertial pose, return null if doesn't exist
  InertialPoseVariables* inertialPose_find(int64_t rigIndex);

  // get inertial pose, return null if doesn't exist (overload)
  const InertialPoseVariables* inertialPose_find(int64_t rigIndex) const;

  // return T_bodyImu_world WHEN a prescribed camera row is acquired
  // if not rolling shutter/time offset: = inertialPose(rigIndex).T_bodyImu_world.value
  // otherwise, time offset corresponding to image row is applied.
  SE3 T_bodyImu_world_atImageRow(int64_t rigIndex, int cameraIndex, float imageRow) const;

  // init new world gravity variable
  void gravityWorld_initNew(const GravityData& gravityWorld, bool constant = true);

  // set world gravity to refer a shared variable
  void gravityWorld_setRef(const std::shared_ptr<GravityVariable>& gravityWorldVar);

  // get world gravity
  GravityVariable& gravityWorld();

  // get world gravity (overload)
  const GravityVariable& gravityWorld() const;

  // add new camera model, return index
  int64_t cameraModel_addNew(
      CameraModelParam&& data,
      int64_t averageTimestampUs,
      int64_t prevVarIndex,
      int sensorIndex,
      bool constant = false);

  // get camera model from index
  CameraModelVariable& cameraModel_at(int64_t index);

  // get camera model from index
  const CameraModelVariable& cameraModel_at(int64_t index) const;

  // get map {(rig,cam) -> camera model index}
  const RigSensorToIndex& rigCamToModelIndex() const;

  // get index of camera model for (rig, cam)
  void cameraModel_setRef(int64_t rigIndex, int s, int index);

  // get camera model
  CameraModelVariable& cameraModel(int64_t rigIndex, int s);

  // get camera model (overload)
  const CameraModelVariable& cameraModel(int64_t rigIndex, int s) const;

  // get camera model, return null if doesn't exist
  CameraModelVariable* cameraModel_find(int64_t rigIndex, int s);

  // get camera model, return null if doesn't exist (overload)
  const CameraModelVariable* cameraModel_find(int64_t rigIndex, int s) const;

  // add new camera extrinsics, return index
  int64_t T_Cam_BodyImu_addNew(
      const SE3& data,
      int64_t averageTimestampUs,
      int64_t prevVarIndex,
      int sensorIndex,
      bool constant = false);

  // get camera extrinsics from index
  ExtrinsicsVariable& T_Cam_BodyImu_at(int64_t index);

  // get camera extrinsics from index (overload)
  const ExtrinsicsVariable& T_Cam_BodyImu_at(int64_t index) const;

  // get map {(rig,cam) -> camera extrinsics index}
  const RigSensorToIndex& rigCamToExtrIndex() const;

  // get index of camera extrinsics for (rig, cam)
  void T_Cam_BodyImu_setRef(int64_t rigIndex, int s, int index);

  // get camera extrinsics
  ExtrinsicsVariable& T_Cam_BodyImu(int64_t rigIndex, int s);

  // get camera extrinsics (overload)
  const ExtrinsicsVariable& T_Cam_BodyImu(int64_t rigIndex, int s) const;

  // get camera extrinsics, return null if doesn't exist
  const ExtrinsicsVariable* T_Cam_BodyImu_find(int64_t rigIndex, int s) const;

  // get camera extrinsics, return null if doesn't exist (overload)
  ExtrinsicsVariable* T_Cam_BodyImu_find(int64_t rigIndex, int s);

  // add new imu calibration, return index
  int64_t imuCalib_addNew(
      const ImuCalibParam& data,
      int64_t averageTimestampUs,
      int64_t prevVarIndex,
      int sensorIndex,
      bool constant = false);

  // get imu calibration from index
  ImuCalibrationVariable& imuCalib_at(int index);

  // get imu calibration from index (overload)
  const ImuCalibrationVariable& imuCalib_at(int index) const;

  // get map {(rig,cam) -> imu calib index}
  const RigSensorToIndex& rigImuToCalibIndex() const;

  // get index of imu calibration for (rig, cam)
  void imuCalib_setRef(int64_t rigIndex, int s, int index);

  // get imu calibration
  ImuCalibrationVariable& imuCalib(int64_t rigIndex, int s);

  // get imu calibration (overload)
  const ImuCalibrationVariable& imuCalib(int64_t rigIndex, int s) const;

  // get imu calibration, return null if doesn't exist
  ImuCalibrationVariable* imuCalib_find(int64_t rigIndex, int s);

  // get imu calibration, return null if doesn't exist (overload)
  const ImuCalibrationVariable* imuCalib_find(int64_t rigIndex, int s) const;

  // add new imu extrinsics, return index
  int64_t T_Imu_BodyImu_addNew(
      const SE3& data,
      int64_t averageTimestampUs,
      int64_t prevVarIndex,
      int sensorIndex,
      bool constant = false);

  // get imu extrinsics from index
  ExtrinsicsVariable& T_Imu_BodyImu_at(int64_t index);

  // get imu extrinsics from index (overload)
  const ExtrinsicsVariable& T_Imu_BodyImu_at(int64_t index) const;

  // get map {(rig,cam) -> imu extrinsics index}
  const RigSensorToIndex& rigImuToExtrIndex() const;

  // get index of imu extrinsics for (rig, cam)
  void T_Imu_BodyImu_setRef(int64_t rigIndex, int s, int index);

  // get imu extrinsics
  ExtrinsicsVariable& T_Imu_BodyImu(int64_t rigIndex, int s);

  // get imu extrinsics (overload)
  const ExtrinsicsVariable& T_Imu_BodyImu(int64_t rigIndex, int s) const;

  // get imu extrinsics, return null if doesn't exist
  ExtrinsicsVariable* T_Imu_BodyImu_find(int64_t rigIndex, int s);

  // get imu extrinsics, return null if doesn't exist
  const ExtrinsicsVariable* T_Imu_BodyImu_find(int64_t rigIndex, int s) const;

  // add new point track, return index
  int64_t pointTrack_addNew(const Vec3& pt);

  // get point track from index
  PointVariable& pointTrack(int64_t trackIndex);

  // get point track from index (overload)
  const PointVariable& pointTrack(int64_t trackIndex) const;

  // get soft (robust) loss used for observations
  SoftLossType& reprojErrorLoss();

  // get soft (robust) loss used for observations (overload)
  const SoftLossType& reprojErrorLoss() const;

  // get soft (robust) loss used for observations
  SoftLossType& imuErrorLoss();

  // get soft (robust) loss used for observations (overload)
  const SoftLossType& imuErrorLoss() const;

  // initialize detector biases
  void detectorBiases_init(int numCameras);

  // get detector bias
  Point2DVariable& detectorBias(int camIndex);

  // get detector bias (overload)
  const Point2DVariable& detectorBias(int camIndex) const;

  // get rolling shutter data
  std::unordered_map<int64_t, RollingShutterData>& rigToRSData();

  int64_t averageTimestampOfRigsInRange(int64_t start, int64_t end);

 private:
  // data
  Verbosity verbosity_;

  // optimizer
  small_thing::Optimizer& opt_;

  SoftLossType reprojErrorLoss_{
      kReprojectionErrorHuberLossWidth,
      kReprojectionErrorHuberLossCutoff};
  SoftLossType imuErrorLoss_{kImuErrorHuberLossWidth, kImuErrorHuberLossCutoff};

  // covariances of optimized variables
  std::unordered_map<int64_t, Eigen::MatrixXd> variableCovariances_;

  // variable (reference to) with world gravity, constant unless computing marginal factors
  std::shared_ptr<GravityVariable> gravityWorld_;

  // rigIndex (in recording's frame list) -> variable
  // (note that only a subset of the rigs in the recording might be loaded)
  std::unordered_map<int64_t, InertialPoseVariables> inertialPoses_;

  // map points
  std::vector<PointVariable> pointTracks_;

  // camera intrinsics
  std::vector<CameraModelVariable> cameraModels_; // per camera type, contains a vector
  RigSensorToIndex rigCamToModelIndex_;

  // camera extrinsics
  std::vector<ExtrinsicsVariable> Ts_Cam_BodyImu_;
  RigSensorToIndex rigCamToExtrIndex_;

  // imu calibration
  std::vector<ImuCalibrationVariable> imuCalibs_;
  RigSensorToIndex rigImuToCalibIndex_;

  // imu extrinsics
  std::vector<ExtrinsicsVariable> Ts_Imu_BodyImu_;
  RigSensorToIndex rigImuToExtrIndex_;

  // detector biases
  std::vector<Point2DVariable> detectorBiases_;

  // rolling shutter data
  std::unordered_map<int64_t, RollingShutterData> rigToRSData_;
};

} // namespace visual_inertial_ba
