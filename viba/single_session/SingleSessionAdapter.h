/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <preint/PreIntegration.h>
#include <rolling_shutters/RollingShutterData.h>
#include <session_data/SessionData.h>
#include <small_thing/Optimizer.h>
#include <viba/common/StatsValueContainer.h>
#include <viba/problem/SingleSessionProblem.h>
#include <viba/single_session/Matcher.h>
#include <viba/single_session/TrajectoryBase.h>
#include <optional>

namespace visual_inertial_ba {

struct TriangulationResult {
  Vec3 point;
  int numInlierObservations;
  std::vector<int64_t> inlierObservationIndices;
};

struct InitSettings;

using InertialPoseGenerator = std::function<std::tuple<SE3, Vec3, Vec3>(int64_t)>;

struct TimeStampInterval {
  int64_t fromTimestampUs;
  int64_t upToTimestampUs; // inclusive
};

// class storing the optimization problem
class SingleSessionAdapter {
 public:
  struct ParamInitSettings {
    int64_t rigIndexStart = -1;
    int64_t rigIndexEnd = -1;
    bool poseConstant = false;
    bool velConstant = false;
    bool omegaConstant = false;
    bool poseToGt = false;
    bool velToGt = false;
    bool omegaToGt = false;

    bool camIntrConstant = false;
    bool camExtrConstant = false;
    bool imuCalibConstant = false;
    bool imuExtrConstant = false;
    bool camIntrToFactory = false;
    bool camExtrToFactory = false;
    bool imuCalibToFactory = false;
    bool imuExtrToFactory = false;

    bool estimateReadoutTime = false;
    bool estimateTimeOffset = false;

    ImuCalibrationOptions imuCalibEstimationOptions;
    ImuCalibrationOptions imuCalibSetToFactory;

    void setTrajectoryConstants(const std::string& args);
    void setTrajectoryToGt(const std::string& args);
    void setCalibConstants(const std::string& args);
    void setCalibToFactory(const std::string& args);
    void setImuCalibEstimationOptions(const std::string& args);
    void setImuCalibToFactory(const std::string& args);

    void printImuCalibOptions() const;
  };

  SingleSessionAdapter(
      SingleSessionProblem& prob,
      const SessionData& fData,
      const Matcher& matcher,
      Verbosity verbosity = Normal);

  void initAllVariablesAndFactors(
      const InitSettings& settings,
      int64_t rigIndexStart = -1,
      int64_t rigIndexEnd = -1,
      const TrajectoryBase* trajectory = nullptr);

  // Initialize all params, can specify a range of rigs
  void initAllParams(const ParamInitSettings& settings);

  // Init from a subset of rigs with determined pose. Poses of rigs not in the subset are
  // interpolated. The rigs loaded are [first - rigWindowGrow, last + rigWindowGrow].
  void initAllParamsInterpolatingRigPoses(
      const ParamInitSettings& settings,
      const std::vector<TimeStampInterval>& timeIntervals,
      const std::vector<int64_t>& krTimestampsUs,
      const std::vector<Sophus::SE3d>& Ts_kr_world,
      const TrajectoryBase* trajectory, // only to possibly set velicites to GT values
      const std::shared_ptr<GravityVariable>& gravityWorld,
      StatsValueContainer* frameDistortionStats_relRot = nullptr,
      StatsValueContainer* frameDistortionStats_relTr = nullptr,
      int rigWindowGrow = 10);

  void initAllParamsWithGtTrajectory(
      const ParamInitSettings& settings,
      const std::vector<TimeStampInterval>& timeIntervals,
      const std::vector<int64_t>& krTimestampsUs,
      const TrajectoryBase* trajectory,
      const std::shared_ptr<GravityVariable>& gravityWorld,
      int rigWindowGrow = 10);

  void initAllParamsWithGtTrajectory(
      const ParamInitSettings& settings,
      const TrajectoryBase* trajectory);

  void addVisualFactors(
      double trackingObsLossRadius,
      double trackingObsLossCutoff,
      bool optimizeDetectorBias);

  void addInertialFactors(double imuLossRadius, double imuLossCutoff);

  // Add priors on angular velocities coming from IMU's preintegrations
  void addOmegaPriors();

  // Random-walk connecting factors
  void addAllRandomWalkFactors(
      double imuRWinflate = 1.0,
      double camIntrRWinflate = 1.0,
      double imuExtrRWinflate = 1.0,
      double camExtrRWinflate = 1.0);

  void addImuRWFactors(double imuRWweight);

  void addCamIntrinsicsRWFactors(double camIntrRWweight);

  void addImuExtrinsicsRWFactors(double imuExtrRWweight);

  void addCamExtrinsicsRWFactors(double camExtrRWweight);

  void addCamIntrFactoryCalibPriors(double stdDevInflate);

  void addImuFactoryCalibPriors(double stdDevInflate);

  void addCamExtrFactoryCalibPriors(double stdDevInflate);

  void addImuExtrFactoryCalibPriors(double stdDevInflate);

  int64_t usedRecordingLengthUs() const;

  void compareCalibrationVsFactory(std::map<std::string, StatsValueContainer>& allStats) const;

  static void showCalibrationCompResults(
      const std::map<std::string, StatsValueContainer>& allStats);

  int numImus() const {
    return numImus_;
  }

  int numCameras() const {
    return numCameras_;
  }

  SingleSessionProblem& getProblem();

  const SingleSessionProblem& getProblem() const;

  void generatePreintegration(std::optional<int> maybePrevRigIndex, int nextRigIndex, int imuIndex);

  int64_t regenerateAllPreintegrationsFromImuMeasurements();

  void updateRollingShutterData();

 private:
  using RigSensorToIndex = std::unordered_map<RigSensorIndices, int64_t, pair_hash>;

  struct Range;
  struct KeyRigInitRef;

  // grow up, stopping if there is no odometry connection
  int64_t growUp(int64_t rigIndex, int rigWindowGrow) const;

  // grow down, stopping if there is no odometry connection
  int64_t growDown(int64_t rigIndex, int rigWindowGrow) const;

  // return true if there is a missin odometry edge in the interval
  bool anyBreakInRange(int64_t a, int64_t b) const;

  KeyRigInitRef computeKeyRigInitRef(
      const std::vector<TimeStampInterval>& timeIntervals,
      const std::vector<int64_t>& krTimestampsUs,
      int rigWindowGrow);

  void initRigsFromGtTrajectory(
      const std::vector<Range>& ranges,
      const TrajectoryBase* trajectory,
      bool poseToGt,
      bool velToGt,
      bool omegaToGt);

  void initNonRigParams(const ParamInitSettings& settings);

  void initRigs(int64_t rigIndexStart, int64_t rigIndexEnd);

  void initRigsInterpolatingPoses(
      const std::vector<TimeStampInterval>& timeIntervals,
      const std::vector<int64_t>& krTimestampsUs,
      const std::vector<SE3>& Ts_kr_world,
      const TrajectoryBase* trajectory,
      bool velToGt,
      bool omegaToGt,
      StatsValueContainer* frameDistortionStats_relRot,
      StatsValueContainer* frameDistortionStats_relTr,
      int rigWindowGrow);

  double walkedDistance(int startFrame, int finalFrame);

  void initRigsFromGtTrajectory(
      int64_t rigIndexStart,
      int64_t rigIndexEnd,
      const TrajectoryBase* trajectory,
      bool poseToGt,
      bool velToGt,
      bool omegaToGt);

  void initRigsFromGtTrajectory(
      const std::vector<TimeStampInterval>& timeIntervals,
      const std::vector<int64_t>& krTimestampsUs,
      const TrajectoryBase* trajectory,
      bool poseToGt,
      bool velToGt,
      bool omegaToGt,
      int rigWindowGrow);

  template <typename LossType>
  void refineTriangulationResult(
      TriangulationResult& result,
      const std::vector<int64_t>& obsIndices,
      double outlierThreshold,
      bool skipOutliers,
      int maxNumIterations,
      const LossType& loss);

  std::optional<TriangulationResult> triangulatePoint(
      const std::vector<int64_t>& obsIndices,
      int seed);

  void initPointsFromObservations();

  void initCamIntrinsics(
      bool constant,
      bool estimateReadoutTime,
      bool estimateTimeOffset,
      bool useFactory);

  void initCamExtrinsics(bool constant, bool useFactory);

  void initImuCalibs(
      bool constant,
      bool useFactory,
      const ImuCalibrationOptions& argImuCalibEstimationOptions,
      const ImuCalibrationOptions& argImuCalibSetToFactor);

  void initImuExtrinsics(bool constant, bool useFactory);

  // returns a list [a1, a2, a3, ..., aZ] so that [aI..a{I+1}) is a list of rig indices
  // notice: not all rig indices in [aI..a{I+1}) might exist in the SingleSessionProblem
  std::vector<int64_t> rigWindowsOfTimeLengthAtMost(double maxTimeLengthSec);

  void filterPointObservations(
      std::vector<int64_t>& filteredObsIndices,
      const std::vector<int64_t>& obsIndices);

  std::map<int64_t, double> computeLogScalings(
      const std::vector<int64_t>& krTimestampsUs,
      const std::vector<SE3>& Ts_kr_world,
      std::vector<int> rigIndices);

  static double scalingAtTimestamp(const std::map<int64_t, double>& logScalings, int64_t timestamp);

  // rolling shutter methods
  void initRollingShutterData(
      const std::unordered_map<int64_t, double>& rigToMaxCameraReadoutTimeSec,
      const std::vector<ImuMeasurement>& imu0Measurements);

  // data
  Verbosity verbosity_;

  // all problem variables
  SingleSessionProblem& prob_;

  // single session data
  const SessionData& fData_;
  const Matcher& matcher_;
  ImuCalibrationOptions imuCalibEstimationOptions;
  ImuCalibrationJacobianIndices imuCalibJacobianIndices;
  const int numRigsInRecording_;
  const int numCameras_;
  const int numImus_;

  // persisted preintegrations (combined, or recomputed)
  std::unordered_map<RigSensorIndices, PreIntegrationData, pair_hash> recomputedPreInts_;

  // pointer to tracking point observations
  std::vector<std::vector<int64_t>> pointTrackObservations_;

  friend class T3MapAdapter;
};

} // namespace visual_inertial_ba
