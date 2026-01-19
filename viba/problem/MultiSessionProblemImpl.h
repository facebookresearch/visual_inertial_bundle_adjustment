/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <viba/problem/MultiSessionProblem.h>

#include <logging/Log.h>
#include <viba/common/Enumerate.h>
#include <viba/problem/BaseMapVisualFactor.h>

namespace visual_inertial_ba {

template <typename KeyRigId, typename MapPointId>
void MultiSessionProblem<KeyRigId, MapPointId>::addBaseMapVisualFactor(
    const BaseMapKeyRig& bmKr,
    int64_t cameraIndex,
    PointVariable& pointTrackVar,
    const Vec2& projBaseRes,
    const Mat22& sqrtH_BaseRes) {
  const auto [factorStorePtr, factorIndex] = opt_.addFactor(
      BaseMapVisualFactor(bmKr, cameraIndex, projBaseRes, sqrtH_BaseRes),
      reprojErrorLoss_, // soft loss (Huber/Cauchy/etc...)
      pointTrackVar);

  if (verbosity_ == Verbose) {
    auto err = factorStorePtr->rawError(factorIndex);
    XR_LOGCI_IF(
        err.has_value(),
        "ViBa::MultiSessionProblem",
        "KR/CAM: [{}:{}], RES: {}",
        &bmKr,
        cameraIndex,
        err.value().transpose());
  }
}

template <typename KeyRigId, typename MapPointId>
MultiSessionProblem<KeyRigId, MapPointId>::MultiSessionProblem(
    small_thing::Optimizer& opt,
    Verbosity verbosity)
    : verbosity_(verbosity), opt_(opt) {}

template <typename KeyRigId, typename MapPointId>
void MultiSessionProblem<KeyRigId, MapPointId>::registerPointVariables() {
  // register recording's tracking points
  for (auto& tProb : tProbs_) {
    tProb.problem->registerPointVariables();
  }

  // register global loop-closing points
  for (auto& [_, ptVar] : globalPointVariables_) {
    opt_.registerVariable(ptVar);
  }
}

template <typename KeyRigId, typename MapPointId>
int64_t MultiSessionProblem<KeyRigId, MapPointId>::numRigVariables() const {
  int64_t count = 0;
  for (const auto& tProb : tProbs_) {
    count += tProb.problem->numRigVariables();
  }
  return count;
}

template <typename KeyRigId, typename MapPointId>
int64_t MultiSessionProblem<KeyRigId, MapPointId>::numCameraModelVariables() const {
  int64_t count = 0;
  for (const auto& tProb : tProbs_) {
    count += tProb.problem->numCameraModelVariables();
  }
  return count;
}

template <typename KeyRigId, typename MapPointId>
int64_t MultiSessionProblem<KeyRigId, MapPointId>::numExtrinsicsVariables() const {
  int64_t count = 0;
  for (const auto& tProb : tProbs_) {
    count += tProb.problem->numExtrinsicsVariables();
  }
  return count;
}

template <typename KeyRigId, typename MapPointId>
int64_t MultiSessionProblem<KeyRigId, MapPointId>::numImuCalibVariables() const {
  int64_t count = 0;
  for (const auto& tProb : tProbs_) {
    count += tProb.problem->numImuCalibVariables();
  }
  return count;
}

template <typename KeyRigId, typename MapPointId>
int64_t MultiSessionProblem<KeyRigId, MapPointId>::numPointTrackVariables() const {
  int64_t count = 0;
  for (const auto& tProb : tProbs_) {
    count += tProb.problem->numPointTrackVariables();
  }
  return count;
}

template <typename KeyRigId, typename MapPointId>
int64_t MultiSessionProblem<KeyRigId, MapPointId>::numGlobalPointVariables() const {
  return globalPointVariables_.size();
}

template <typename KeyRigId, typename MapPointId>
int64_t MultiSessionProblem<KeyRigId, MapPointId>::numBaseMapKeyRigs() const {
  return baseMapKeyRigs_.size();
}

template <typename KeyRigId, typename MapPointId>
void MultiSessionProblem<KeyRigId, MapPointId>::trackingProblems_init(int numTrackingProblems) {
  for (int i = 0; i < numTrackingProblems; i++) {
    tProbs_.push_back({.problem = std::make_unique<SingleSessionProblem>(opt_)});
  }
}

template <typename KeyRigId, typename MapPointId>
void MultiSessionProblem<KeyRigId, MapPointId>::trackingProblem_setLabel(
    int recIndex,
    const std::string& label) {
  tProbs_[recIndex].label = label;
}

template <typename KeyRigId, typename MapPointId>
SingleSessionProblem& MultiSessionProblem<KeyRigId, MapPointId>::trackingProblem(int recIndex) {
  return *tProbs_[recIndex].problem;
}

template <typename KeyRigId, typename MapPointId>
const SingleSessionProblem& MultiSessionProblem<KeyRigId, MapPointId>::trackingProblem(
    int recIndex) const {
  return *tProbs_[recIndex].problem;
}

template <typename KeyRigId, typename MapPointId>
SoftLossType& MultiSessionProblem<KeyRigId, MapPointId>::reprojErrorLoss() {
  return reprojErrorLoss_;
}

template <typename KeyRigId, typename MapPointId>
const SoftLossType& MultiSessionProblem<KeyRigId, MapPointId>::reprojErrorLoss() const {
  return reprojErrorLoss_;
}

template <typename KeyRigId, typename MapPointId>
void MultiSessionProblem<KeyRigId, MapPointId>::gravityWorld_initNew(
    const GravityData& gravityWorld,
    bool constant) {
  gravityWorld_ = std::make_shared<GravityVariable>(gravityWorld);
  gravityWorld_->setConstant(constant);
}

template <typename KeyRigId, typename MapPointId>
GravityVariable& MultiSessionProblem<KeyRigId, MapPointId>::gravityWorld() {
  XR_CHECK(gravityWorld_);
  return *gravityWorld_;
}

template <typename KeyRigId, typename MapPointId>
const GravityVariable& MultiSessionProblem<KeyRigId, MapPointId>::gravityWorld() const {
  XR_CHECK(gravityWorld_);
  return *gravityWorld_;
}

template <typename KeyRigId, typename MapPointId>
std::shared_ptr<GravityVariable>
MultiSessionProblem<KeyRigId, MapPointId>::gravityWorld_sharedVar() {
  return gravityWorld_;
}

template <typename KeyRigId, typename MapPointId>
void MultiSessionProblem<KeyRigId, MapPointId>::detectorBiases_init(int recIndex, int numCameras) {
  tProbs_[recIndex].detectorBiases.assign(numCameras, Vec2::Zero());
}

template <typename KeyRigId, typename MapPointId>
Point2DVariable& MultiSessionProblem<KeyRigId, MapPointId>::detectorBias(
    int recIndex,
    int camIndex) {
  return tProbs_[recIndex].detectorBiases[camIndex];
}

template <typename KeyRigId, typename MapPointId>
const Point2DVariable& MultiSessionProblem<KeyRigId, MapPointId>::detectorBias(
    int recIndex,
    int camIndex) const {
  return tProbs_[recIndex].detectorBiases[camIndex];
}

template <typename KeyRigId, typename MapPointId>
void MultiSessionProblem<KeyRigId, MapPointId>::globalPoint_set(
    const MapPointId& id,
    const Vec3& pt) {
  globalPointVariables_[id] = pt;
}

template <typename KeyRigId, typename MapPointId>
PointVariable* MultiSessionProblem<KeyRigId, MapPointId>::globalPoint_find(const MapPointId& id) {
  auto it = globalPointVariables_.find(id);
  return it != globalPointVariables_.end() ? &it->second : nullptr;
}

template <typename KeyRigId, typename MapPointId>
BaseMapKeyRig& MultiSessionProblem<KeyRigId, MapPointId>::baseMapKeyRig_add(const KeyRigId& id) {
  return baseMapKeyRigs_[id];
}

template <typename KeyRigId, typename MapPointId>
BaseMapKeyRig* MultiSessionProblem<KeyRigId, MapPointId>::baseMapKeyRig_find(const KeyRigId& id) {
  auto it = baseMapKeyRigs_.find(id);
  return it != baseMapKeyRigs_.end() ? &it->second : nullptr;
}

template <typename KeyRigId, typename MapPointId>
void MultiSessionProblem<KeyRigId, MapPointId>::printDetectorBiases() {
  for (auto& tProb : tProbs_) {
    tProb.problem->printDetectorBiases(tProb.label + " - tracking points");
    SingleSessionProblem::printDetectorBiases(
        tProb.label + " - global map points", tProb.detectorBiases);
  }
}

template <typename KeyRigId, typename MapPointId>
void MultiSessionProblem<KeyRigId, MapPointId>::applyWorldTransformation(const SE3& T_newW_world) {
  for (auto& [_, ptVar] : globalPointVariables_) {
    ptVar.value = T_newW_world * ptVar.value;
  }

  SE3 T_world_newW = T_newW_world.inverse();
  for (auto& [_, baseRigVar] : baseMapKeyRigs_) {
    baseRigVar.T_bodyImu_world = baseRigVar.T_bodyImu_world * T_world_newW;
  }

  gravityWorld_->value.vec = T_newW_world.so3() * gravityWorld_->value.vec;

  for (auto& tProb : tProbs_) {
    tProb.problem->applyWorldTransformation(T_newW_world);
  }
}

} // namespace visual_inertial_ba
