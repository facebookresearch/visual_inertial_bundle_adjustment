/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
#include <viba/common/Enumerate.h>
#include <viba/problem/Histograms.h>
#include <viba/problem/SingleSessionProblem.h>

#include <logging/Checks.h>
#define DEFAULT_LOG_CHANNEL "ViBa::SingleSessionProblem"
#include <logging/Log.h>

namespace visual_inertial_ba {

SingleSessionProblem::SingleSessionProblem(small_thing::Optimizer& opt, Verbosity verbosity)
    : verbosity_(verbosity), opt_(opt) {}

void SingleSessionProblem::setAllRigsConstant(bool constPose, bool constLinVel, bool constAngVel) {
  if (verbosity_ != Muted) {
    XR_LOGI(
        "Setting {} inertial rigs constness: (constPose: {}, constLinVel: {}, constAngVel: {})",
        inertialPoses_.size(),
        constPose,
        constLinVel,
        constAngVel);
  }
  for (auto& [_, inertialPose] : inertialPoses_) {
    XR_CHECK(!inertialPose.T_bodyImu_world.isRegistered());
    XR_CHECK(!inertialPose.vel_world.isRegistered());
    XR_CHECK(!inertialPose.omega.isRegistered());
    inertialPose.T_bodyImu_world.setConstant(constPose);
    inertialPose.vel_world.setConstant(constLinVel);
    inertialPose.omega.setConstant(constAngVel);
  }
}

void SingleSessionProblem::registerPointVariables() {
  for (auto& ptVar : pointTracks_) {
    opt_.registerVariable(ptVar);
  }
}

std::optional<int64_t> SingleSessionProblem::firstRigIndex() const {
  std::optional<int64_t> firstRigIndex;
  for (const auto& [rigIndex, _] : inertialPoses_) {
    if (!firstRigIndex.has_value() || (rigIndex < *firstRigIndex)) {
      firstRigIndex = rigIndex;
    }
  }
  return firstRigIndex;
}

std::vector<int64_t> SingleSessionProblem::sortedRigIndices() const {
  std::vector<int64_t> rigIndices;
  for (const auto& [rigIndex, _] : inertialPoses_) {
    rigIndices.push_back(rigIndex);
  }
  std::sort(rigIndices.begin(), rigIndices.end());
  return rigIndices;
}

void SingleSessionProblem::computeCovariances() {
  const auto maybeFirstRigIndex = firstRigIndex();
  if (!maybeFirstRigIndex.has_value()) {
    return;
  }

  // make sure position and yaw of first rig are constained
  XR_CHECK_EQ(
      clearPosePriors(), 0, "Computation of covariances: problem should have no pose priors");
  constrainPositionAndYaw(*maybeFirstRigIndex);

  // indices for computation of covariance, need optimizer indices and rig index for reference
  std::unordered_set<int64_t> nonRigCovVarIndices;
  for (const auto& imuVar : imuCalibs_) {
    XR_CHECK_GE(imuVar.var.index, 0); // TODO: allow set to constant
    nonRigCovVarIndices.insert(imuVar.var.index);
  }
  for (const auto& extrVar : Ts_Imu_BodyImu_) {
    XR_CHECK_GE(extrVar.var.index, 0); // TODO: allow set to constant
    nonRigCovVarIndices.insert(extrVar.var.index);
  }
  for (const auto& intrVar : cameraModels_) {
    XR_CHECK_GE(intrVar.var.index, 0); // TODO: allow set to constant
    nonRigCovVarIndices.insert(intrVar.var.index);
  }
  for (const auto& extrVar : Ts_Cam_BodyImu_) {
    XR_CHECK_GE(extrVar.var.index, 0); // TODO: allow set to constant
    nonRigCovVarIndices.insert(extrVar.var.index);
  }

  bool estimatingOmega = findOrDie(inertialPoses_, *maybeFirstRigIndex).omega.isRegistered();

  std::vector<std::vector<int64_t>> covVarIndices;
  for (const auto& [rigIndex, var] : inertialPoses_) {
    XR_CHECK(var.T_bodyImu_world.isRegistered());
    XR_CHECK(var.vel_world.isRegistered());
    XR_CHECK_EQ(estimatingOmega, var.omega.isRegistered());

    // add all variables for the rig (pose, velocity, etc)
    covVarIndices.push_back({var.T_bodyImu_world.index, var.vel_world.index});
    if (estimatingOmega) {
      covVarIndices.back().push_back(var.omega.index);
    }
  }

  // add the non-rig variables, made unique
  for (int64_t index : nonRigCovVarIndices) {
    covVarIndices.push_back({index});
  }

  if (verbosity_ != Muted) {
    XR_LOGI(
        "Computing covs... ({} variables: {} rigs, {} imu calibs, {} imu extr, {} cam intr, {} cam extr)",
        covVarIndices.size(),
        inertialPoses_.size(),
        imuCalibs_.size(),
        Ts_Imu_BodyImu_.size(),
        cameraModels_.size(),
        Ts_Cam_BodyImu_.size());
  }
  auto covAndOffsets = opt_.computeJointCovariances({.damping = 0}, covVarIndices);
  XR_CHECK_EQ(covAndOffsets.size(), covVarIndices.size());

  // remove the pose prior we added, to leave the problem in a "clean" state
  XR_CHECK_EQ(clearPosePriors(), 1);

  if (verbosity_ != Muted) {
    XR_LOGI("Done!");
  }
  for (auto [i, varIndices] : enumerate(covVarIndices)) {
    variableCovariances_[varIndices[0]] = std::move(covAndOffsets[i].first);
  }
}

int64_t SingleSessionProblem::numRigVariables() const {
  return inertialPoses_.size();
}

int64_t SingleSessionProblem::numCameraModelVariables() const {
  return cameraModels_.size();
}

int64_t SingleSessionProblem::numExtrinsicsVariables() const {
  return Ts_Cam_BodyImu_.size() + Ts_Imu_BodyImu_.size();
}

int64_t SingleSessionProblem::numCameraExtrinsicsVariables() const {
  return Ts_Cam_BodyImu_.size();
}

int64_t SingleSessionProblem::numImuExtrinsicsVariables() const {
  return Ts_Imu_BodyImu_.size();
}

int64_t SingleSessionProblem::numImuCalibVariables() const {
  return imuCalibs_.size();
}

int64_t SingleSessionProblem::numPointTrackVariables() const {
  return pointTracks_.size();
}

const std::unordered_map<int64_t, Eigen::MatrixXd>& SingleSessionProblem::variableCovariances()
    const {
  return variableCovariances_;
}

const std::unordered_map<int64_t, InertialPoseVariables>& SingleSessionProblem::inertialPoses()
    const {
  return inertialPoses_;
}

bool SingleSessionProblem::inertialPose_exists(int64_t rigIndex) {
  return inertialPoses_.count(rigIndex) > 0;
}

void SingleSessionProblem::inertialPose_set(
    int64_t rigIndex,
    const SE3& T_bodyImu_world,
    const Vec3& vel_world,
    const Vec3& omega,
    int64_t timestampUs) {
  inertialPoses_[rigIndex] = {
      .T_bodyImu_world = T_bodyImu_world,
      .vel_world = vel_world,
      .omega = omega,
      .timestampUs = timestampUs,
  };
}

InertialPoseVariables& SingleSessionProblem::inertialPose(int64_t rigIndex) {
  return findOrDie(inertialPoses_, rigIndex);
}

const InertialPoseVariables& SingleSessionProblem::inertialPose(int64_t rigIndex) const {
  return findOrDie(inertialPoses_, rigIndex);
}

InertialPoseVariables* SingleSessionProblem::inertialPose_find(int64_t rigIndex) {
  const auto it = inertialPoses_.find(rigIndex);
  return it != inertialPoses_.end() ? &it->second : nullptr;
}

const InertialPoseVariables* SingleSessionProblem::inertialPose_find(int64_t rigIndex) const {
  const auto it = inertialPoses_.find(rigIndex);
  return it != inertialPoses_.end() ? &it->second : nullptr;
}

void SingleSessionProblem::gravityWorld_initNew(const GravityData& gravityWorld, bool constant) {
  gravityWorld_ = std::make_shared<GravityVariable>(gravityWorld);
  gravityWorld_->setConstant(constant);
}

void SingleSessionProblem::gravityWorld_setRef(
    const std::shared_ptr<GravityVariable>& gravityWorldVar) {
  gravityWorld_ = gravityWorldVar;
}

GravityVariable& SingleSessionProblem::gravityWorld() {
  XR_CHECK(gravityWorld_);
  return *gravityWorld_;
}

const GravityVariable& SingleSessionProblem::gravityWorld() const {
  XR_CHECK(gravityWorld_);
  return *gravityWorld_;
}

int64_t SingleSessionProblem::cameraModel_addNew(
    CameraModelParam&& data,
    int64_t averageTimestampUs,
    int64_t prevVarIndex,
    int sensorIndex,
    bool constant) {
  cameraModels_.push_back(
      CameraModelVariable{
          .var = std::move(data),
          .averageTimestampUs = averageTimestampUs,
          .prevVarIndex = prevVarIndex,
          .sensorIndex = sensorIndex,
      });
  cameraModels_.back().var.setConstant(constant);
  return cameraModels_.size() - 1;
}

CameraModelVariable& SingleSessionProblem::cameraModel_at(int64_t index) {
  return cameraModels_[index];
}

const CameraModelVariable& SingleSessionProblem::cameraModel_at(int64_t index) const {
  return cameraModels_[index];
}

const SingleSessionProblem::RigSensorToIndex& SingleSessionProblem::rigCamToModelIndex() const {
  return rigCamToModelIndex_;
}

void SingleSessionProblem::cameraModel_setRef(int64_t rigIndex, int s, int index) {
  rigCamToModelIndex_[{rigIndex, s}] = index;
}

CameraModelVariable& SingleSessionProblem::cameraModel(int64_t rigIndex, int s) {
  return cameraModels_[findOrDie(rigCamToModelIndex_, {rigIndex, s})];
}

const CameraModelVariable& SingleSessionProblem::cameraModel(int64_t rigIndex, int s) const {
  return cameraModels_[findOrDie(rigCamToModelIndex_, {rigIndex, s})];
}

CameraModelVariable* SingleSessionProblem::cameraModel_find(int64_t rigIndex, int s) {
  const auto it = rigCamToModelIndex_.find({rigIndex, s});
  return it != rigCamToModelIndex_.end() ? &cameraModels_[it->second] : nullptr;
}

const CameraModelVariable* SingleSessionProblem::cameraModel_find(int64_t rigIndex, int s) const {
  const auto it = rigCamToModelIndex_.find({rigIndex, s});
  return it != rigCamToModelIndex_.end() ? &cameraModels_[it->second] : nullptr;
}

int64_t SingleSessionProblem::T_Cam_BodyImu_addNew(
    const SE3& data,
    int64_t averageTimestampUs,
    int64_t prevVarIndex,
    int sensorIndex,
    bool constant) {
  Ts_Cam_BodyImu_.push_back(
      ExtrinsicsVariable{
          .var = data,
          .averageTimestampUs = averageTimestampUs,
          .prevVarIndex = prevVarIndex,
          .sensorIndex = sensorIndex,
      });
  Ts_Cam_BodyImu_.back().var.setConstant(constant);
  return Ts_Cam_BodyImu_.size() - 1;
}

ExtrinsicsVariable& SingleSessionProblem::T_Cam_BodyImu_at(int64_t index) {
  return Ts_Cam_BodyImu_[index];
}

const ExtrinsicsVariable& SingleSessionProblem::T_Cam_BodyImu_at(int64_t index) const {
  return Ts_Cam_BodyImu_[index];
}

const SingleSessionProblem::RigSensorToIndex& SingleSessionProblem::rigCamToExtrIndex() const {
  return rigCamToExtrIndex_;
}

void SingleSessionProblem::T_Cam_BodyImu_setRef(int64_t rigIndex, int s, int index) {
  rigCamToExtrIndex_[{rigIndex, s}] = index;
}

ExtrinsicsVariable& SingleSessionProblem::T_Cam_BodyImu(int64_t rigIndex, int s) {
  return Ts_Cam_BodyImu_[findOrDie(rigCamToExtrIndex_, {rigIndex, s})];
}

const ExtrinsicsVariable& SingleSessionProblem::T_Cam_BodyImu(int64_t rigIndex, int s) const {
  return Ts_Cam_BodyImu_[findOrDie(rigCamToExtrIndex_, {rigIndex, s})];
}

ExtrinsicsVariable* SingleSessionProblem::T_Cam_BodyImu_find(int64_t rigIndex, int s) {
  const auto it = rigCamToExtrIndex_.find({rigIndex, s});
  return it != rigCamToExtrIndex_.end() ? &Ts_Cam_BodyImu_[it->second] : nullptr;
}

const ExtrinsicsVariable* SingleSessionProblem::T_Cam_BodyImu_find(int64_t rigIndex, int s) const {
  const auto it = rigCamToExtrIndex_.find({rigIndex, s});
  return it != rigCamToExtrIndex_.end() ? &Ts_Cam_BodyImu_[it->second] : nullptr;
}

int64_t SingleSessionProblem::imuCalib_addNew(
    const ImuCalibParam& data,
    int64_t averageTimestampUs,
    int64_t prevVarIndex,
    int sensorIndex,
    bool constant) {
  imuCalibs_.push_back(
      ImuCalibrationVariable{
          .var = data,
          .averageTimestampUs = averageTimestampUs,
          .prevVarIndex = prevVarIndex,
          .sensorIndex = sensorIndex,
      });
  imuCalibs_.back().var.setConstant(constant);
  return imuCalibs_.size() - 1;
}

ImuCalibrationVariable& SingleSessionProblem::imuCalib_at(int index) {
  return imuCalibs_[index];
}

const ImuCalibrationVariable& SingleSessionProblem::imuCalib_at(int index) const {
  return imuCalibs_[index];
}

const SingleSessionProblem::RigSensorToIndex& SingleSessionProblem::rigImuToCalibIndex() const {
  return rigImuToCalibIndex_;
}

void SingleSessionProblem::imuCalib_setRef(int64_t rigIndex, int s, int index) {
  rigImuToCalibIndex_[{rigIndex, s}] = index;
}

ImuCalibrationVariable& SingleSessionProblem::imuCalib(int64_t rigIndex, int s) {
  return imuCalibs_[findOrDie(rigImuToCalibIndex_, {rigIndex, s})];
}

const ImuCalibrationVariable& SingleSessionProblem::imuCalib(int64_t rigIndex, int s) const {
  return imuCalibs_[findOrDie(rigImuToCalibIndex_, {rigIndex, s})];
}

ImuCalibrationVariable* SingleSessionProblem::imuCalib_find(int64_t rigIndex, int s) {
  const auto it = rigImuToCalibIndex_.find({rigIndex, s});
  return it != rigImuToCalibIndex_.end() ? &imuCalibs_[it->second] : nullptr;
}

const ImuCalibrationVariable* SingleSessionProblem::imuCalib_find(int64_t rigIndex, int s) const {
  const auto it = rigImuToCalibIndex_.find({rigIndex, s});
  return it != rigImuToCalibIndex_.end() ? &imuCalibs_[it->second] : nullptr;
}

int64_t SingleSessionProblem::T_Imu_BodyImu_addNew(
    const SE3& data,
    int64_t averageTimestampUs,
    int64_t prevVarIndex,
    int sensorIndex,
    bool constant) {
  Ts_Imu_BodyImu_.push_back(
      ExtrinsicsVariable{
          .var = data,
          .averageTimestampUs = averageTimestampUs,
          .prevVarIndex = prevVarIndex,
          .sensorIndex = sensorIndex,
      });
  Ts_Imu_BodyImu_.back().var.setConstant(constant);
  return Ts_Imu_BodyImu_.size() - 1;
}

ExtrinsicsVariable& SingleSessionProblem::T_Imu_BodyImu_at(int64_t index) {
  return Ts_Imu_BodyImu_[index];
}

const ExtrinsicsVariable& SingleSessionProblem::T_Imu_BodyImu_at(int64_t index) const {
  return Ts_Imu_BodyImu_[index];
}

const SingleSessionProblem::RigSensorToIndex& SingleSessionProblem::rigImuToExtrIndex() const {
  return rigImuToExtrIndex_;
}

void SingleSessionProblem::T_Imu_BodyImu_setRef(int64_t rigIndex, int s, int index) {
  rigImuToExtrIndex_[{rigIndex, s}] = index;
}

ExtrinsicsVariable& SingleSessionProblem::T_Imu_BodyImu(int64_t rigIndex, int s) {
  return Ts_Imu_BodyImu_[findOrDie(rigImuToExtrIndex_, {rigIndex, s})];
}

const ExtrinsicsVariable& SingleSessionProblem::T_Imu_BodyImu(int64_t rigIndex, int s) const {
  return Ts_Imu_BodyImu_[findOrDie(rigImuToExtrIndex_, {rigIndex, s})];
}

ExtrinsicsVariable* SingleSessionProblem::T_Imu_BodyImu_find(int64_t rigIndex, int s) {
  const auto it = rigImuToExtrIndex_.find({rigIndex, s});
  return it != rigImuToExtrIndex_.end() ? &Ts_Imu_BodyImu_[it->second] : nullptr;
}

const ExtrinsicsVariable* SingleSessionProblem::T_Imu_BodyImu_find(int64_t rigIndex, int s) const {
  const auto it = rigImuToExtrIndex_.find({rigIndex, s});
  return it != rigImuToExtrIndex_.end() ? &Ts_Imu_BodyImu_[it->second] : nullptr;
}

int64_t SingleSessionProblem::pointTrack_addNew(const Vec3& pt) {
  pointTracks_.emplace_back(pt);
  return pointTracks_.size() - 1;
}

PointVariable& SingleSessionProblem::pointTrack(int64_t trackIndex) {
  XR_CHECK_LT(trackIndex, pointTracks_.size());
  return pointTracks_[trackIndex];
}

const PointVariable& SingleSessionProblem::pointTrack(int64_t trackIndex) const {
  XR_CHECK_LT(trackIndex, pointTracks_.size());
  return pointTracks_[trackIndex];
}

SoftLossType& SingleSessionProblem::reprojErrorLoss() {
  return reprojErrorLoss_;
}

const SoftLossType& SingleSessionProblem::reprojErrorLoss() const {
  return reprojErrorLoss_;
}

SoftLossType& SingleSessionProblem::imuErrorLoss() {
  return imuErrorLoss_;
}

const SoftLossType& SingleSessionProblem::imuErrorLoss() const {
  return imuErrorLoss_;
}

void SingleSessionProblem::detectorBiases_init(int numCameras) {
  detectorBiases_.assign(numCameras, Vec2::Zero());
}

Point2DVariable& SingleSessionProblem::detectorBias(int camIndex) {
  return detectorBiases_[camIndex];
}

const Point2DVariable& SingleSessionProblem::detectorBias(int camIndex) const {
  return detectorBiases_[camIndex];
}

std::unordered_map<int64_t, RollingShutterData>& SingleSessionProblem::rigToRSData() {
  return rigToRSData_;
}

// utils: return true if element is address of and element in std::vector
template <typename T, typename V>
static bool belongsToVec(const std::vector<T>& vec, V* el) {
  intptr_t start = (intptr_t)&vec[0], back = (intptr_t)(&vec.back()), elptr = (intptr_t)el;
  return (elptr >= start) && (elptr <= back);
}

bool SingleSessionProblem::isOwnPointVar(small_thing::VarBase* var) const {
  return belongsToVec(pointTracks_, var);
}

bool SingleSessionProblem::isOwnImuCalibVar(small_thing::VarBase* var) const {
  return belongsToVec(imuCalibs_, var);
}

bool SingleSessionProblem::isOwnImuExtrinsicsVar(small_thing::VarBase* var) const {
  return belongsToVec(Ts_Imu_BodyImu_, var);
}

bool SingleSessionProblem::isOwnCamExtrinsicsVar(small_thing::VarBase* var) const {
  return belongsToVec(Ts_Cam_BodyImu_, var);
}

bool SingleSessionProblem::isOwnCameraModelVar(small_thing::VarBase* var) const {
  return belongsToVec(cameraModels_, var);
}

void SingleSessionProblem::showHistogram(bool simpleStats) const {
  Histograms h(opt_);
  if (simpleStats) {
    h.showPixelErrors = false; // image-distance pixel reproj errors
    h.showRotVelPos = false; // separate histograms for rot/vel/pos
    h.separateSecondaryInertial = false; // separate main/secondary imu
    h.showAggregateCalibFactors = true; // one histogram for all rw/fprio factors
  }
  h.show();
};

void SingleSessionProblem::applyWorldTransformation(const SE3& T_newW_world) {
  for (auto& ptVar : pointTracks_) {
    ptVar.value = T_newW_world * ptVar.value;
  }

  SE3 T_world_newW = T_newW_world.inverse();
  for (auto& [_, inertialPoseVar] : inertialPoses_) {
    inertialPoseVar.T_bodyImu_world.value = inertialPoseVar.T_bodyImu_world.value * T_world_newW;
    inertialPoseVar.vel_world.value = T_newW_world.so3() * inertialPoseVar.vel_world.value;
  }

  // apply transformation on gravity only if we are the only owner
  if (gravityWorld_.use_count() == 1) {
    gravityWorld_->value.vec = T_newW_world.so3() * gravityWorld_->value.vec;
  }
}

int64_t SingleSessionProblem::averageTimestampOfRigsInRange(int64_t start, int64_t end) {
  double sumSec = 0.0;
  int nFound = 0;
  for (int64_t i = start; i < end; i++) {
    if (const auto* pRig = inertialPose_find(i)) {
      sumSec += pRig->timestampUs * 1e-6;
      nFound++;
    }
  }
  if (nFound == 0) {
    return -1;
  }
  return (int64_t)((sumSec / nFound) * 1e6);
}

} // namespace visual_inertial_ba
