/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <viba/problem/BaseMapVisualFactor.h>
#include <viba/problem/SingleSessionProblem.h>

namespace visual_inertial_ba {

struct T3InitSettings;

// structure holding referenced to tracking problems inside a mapping problem class
struct SingleSessionProbData {
  std::unique_ptr<SingleSessionProblem> problem;
  std::string label; // used to refer to this problem in pretty-printing
  std::vector<Point2DVariable> detectorBiases;
};

template <typename KeyRigId, typename MapPointId>
class MultiSessionProblem {
 public:
  explicit MultiSessionProblem(small_thing::Optimizer& opt, Verbosity verbosity = Normal);

  // print histogram of the whole problem
  void showHistogram(bool simpleStats, bool separatePerRecordingStats) const;

  // register all point variables in the optimizer (needed for Schur-complement trick)
  void registerPointVariables();

  // number of rig variables
  int64_t numRigVariables() const;

  // number of camera model variables
  int64_t numCameraModelVariables() const;

  // number of extrinsics (imu and camera)
  int64_t numExtrinsicsVariables() const;

  // number of imu calibration variables
  int64_t numImuCalibVariables() const;

  // (total) nummer of point track vraiables, in all trcking problems
  int64_t numPointTrackVariables() const;

  // number of global (map) point variables
  int64_t numGlobalPointVariables() const;

  // number of constant base-map key rigs
  int64_t numBaseMapKeyRigs() const;

  // pretty-print detector biases
  void printDetectorBiases();

  // apply an SE3 transformation to the whole problem (and the tracking problems)
  void applyWorldTransformation(const SE3& T_newW_world);

  // get (populating on request) data for a constant base-map's keyrig
  const BaseMapKeyRig& getBaseMapKeyRig(const KeyRigId& hKr);

  // add a basemap visual factor
  void addBaseMapVisualFactor(
      const BaseMapKeyRig& bmKr,
      int64_t cameraIndex,
      PointVariable& pointTrackVar,
      const Vec2& projBaseRes,
      const Mat22& sqrtH_BaseRes);

  // init tracking problems
  void trackingProblems_init(int numTrackingProblems);

  // set "label" of tracking problem (only used in histogram/detector bias pretty printing)
  void trackingProblem_setLabel(int recIndex, const std::string& label);

  // return n-th tracking problem
  SingleSessionProblem& trackingProblem(int recIndex);

  // return n-th tracking problem (overload)
  const SingleSessionProblem& trackingProblem(int recIndex) const;

  // return soft (robust) loss used for map-point observations
  SoftLossType& reprojErrorLoss();

  // return soft (robust) loss used for map-point observations (overload)
  const SoftLossType& reprojErrorLoss() const;

  // init gravity world variable
  void gravityWorld_initNew(const GravityData& gravityWorld, bool constant = true);

  // get gravity world variable
  GravityVariable& gravityWorld();

  // get gravity world variable (overload)
  const GravityVariable& gravityWorld() const;

  // get shared pointer to (already initialized) gravity world variable
  std::shared_ptr<GravityVariable> gravityWorld_sharedVar();

  // init detector biases used for given recording's map points
  void detectorBiases_init(int recIndex, int numCameras);

  // get detector bias
  Point2DVariable& detectorBias(int recIndex, int camIndex);

  // get detector bias (overload)
  const Point2DVariable& detectorBias(int recIndex, int camIndex) const;

  // create new global point with given id
  void globalPoint_set(const MapPointId& id, const Vec3& pt);

  // find global point with given id (return null if not found)
  PointVariable* globalPoint_find(const MapPointId& id);

  // add base map's key rig with given id, return reference to be populated
  BaseMapKeyRig& baseMapKeyRig_add(const KeyRigId& id);

  // find base map's key rig with given id (return null if not found)
  BaseMapKeyRig* baseMapKeyRig_find(const KeyRigId& id);

 private:
  Verbosity verbosity_;

  // recording-specific data
  std::vector<SingleSessionProbData> tProbs_;

  small_thing::Optimizer& opt_;

  // soft (robust) loss for map point observations
  SoftLossType reprojErrorLoss_{
      kReprojectionErrorHuberLossWidth,
      kReprojectionErrorHuberLossCutoff};

  // variables
  std::shared_ptr<GravityVariable> gravityWorld_;
  std::unordered_map<MapPointId, PointVariable> globalPointVariables_;
  std::unordered_map<KeyRigId, BaseMapKeyRig> baseMapKeyRigs_;
};

} // namespace visual_inertial_ba
