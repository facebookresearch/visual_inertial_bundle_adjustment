/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
#include <sophus/interpolate.hpp>
#include <viba/common/Enumerate.h>
#include <viba/common/StatsValueContainer.h>
#include <viba/problem/Types.h>
#include <viba/single_session/SingleSessionAdapter.h>

#define DEFAULT_LOG_CHANNEL "ViBa::InitRigs"
#include <logging/Log.h>

namespace visual_inertial_ba {

struct SingleSessionAdapter::Range {
  int64_t rigIndexStart;
  int64_t rigIndexEnd;
};

struct SingleSessionAdapter::KeyRigInitRef {
  std::vector<int> rigIndices; // indices of matched rigs
  std::map<int, int> rigIndex_to_krIndex;

  std::vector<Range> ranges;
};

int64_t SingleSessionAdapter::growUp(int64_t rigIndex, int rigWindowGrow) const {
  int64_t i = rigIndex;
  while ((i < rigIndex + rigWindowGrow) && (i < numRigsInRecording_ - 1) &&
         !matcher_.resetRigIndices.count(i)) {
    i++;
  }
  return i;
}

int64_t SingleSessionAdapter::growDown(int64_t rigIndex, int rigWindowGrow) const {
  int64_t i = rigIndex;
  while ((i > rigIndex - rigWindowGrow) && (i > 0) && !matcher_.resetRigIndices.count(i - 1)) {
    i--;
  }
  return i;
}

bool SingleSessionAdapter::anyBreakInRange(int64_t a, int64_t b) const {
  for (int64_t i = a; i < b; i++) {
    if (matcher_.resetRigIndices.count(i)) {
      return true;
    }
  }
  return false;
}

SingleSessionAdapter::KeyRigInitRef SingleSessionAdapter::computeKeyRigInitRef(
    const std::vector<TimeStampInterval>& timeIntervals_,
    const std::vector<int64_t>& krTimestampsUs,
    int rigWindowGrow) {
  XR_CHECK(!krTimestampsUs.empty());

  // sort and merge time intervals, so we can query if keyrigs are in the same interval
  std::vector<TimeStampInterval> timeIntervals = timeIntervals_;
  std::sort(timeIntervals.begin(), timeIntervals.end(), [](const auto& a, const auto& b) -> bool {
    return a.fromTimestampUs < b.fromTimestampUs;
  });
  int64_t previousBlockStart = -1;
  std::map<int64_t, int64_t> intervalFinalToStart; // start/final for merged timestamp ranges
  for (int64_t i = 0; i < timeIntervals.size(); i++) {
    bool isNewStart = (i == 0) ||
        (i > 0 && timeIntervals[i].fromTimestampUs > timeIntervals[i - 1].upToTimestampUs);
    if (isNewStart) {
      if (i > 0) {
        intervalFinalToStart[timeIntervals[i - 1].upToTimestampUs] = previousBlockStart;
      }
      previousBlockStart = timeIntervals[i].fromTimestampUs;
    }
  }
  if (!timeIntervals.empty()) {
    intervalFinalToStart[timeIntervals.back().upToTimestampUs] = previousBlockStart;
  }
  // return interval start or -1
  auto timestampToIntervalFinal = [&](const int64_t timestamp) -> int64_t {
    const auto it = intervalFinalToStart.lower_bound(timestamp);
    return it != intervalFinalToStart.end() && it->second <= timestamp ? it->first : -1;
  };

  // verify that the input vector is sorted
  for (size_t i = 1; i < krTimestampsUs.size(); i++) {
    XR_CHECK_LT(krTimestampsUs[i - 1], krTimestampsUs[i]);
  }

  std::vector<int> rigIndices; // indices of matched rigs
  std::map<int, int> rigIndex_to_krIndex; // map to index in `krTimestampsUs/Ts_bodyImu_world`

  // iterate over provided rig poses (via timestamps). We populate the arrays above
  std::vector<SingleSessionAdapter::Range> ranges{{.rigIndexStart = -1, .rigIndexEnd = -1}};
  int64_t prevIntervalStart = -1;
  for (const auto& [krIndex, krTimestampUs] : enumerate(krTimestampsUs)) {
    const int rigIndex = findOrDie(matcher_.timestampToRigIndex, krTimestampUs);

    const int64_t newIntervalStart = timestampToIntervalFinal(krTimestampUs);
    const bool noSeparateRanges = (newIntervalStart >= 0 && prevIntervalStart == newIntervalStart);

    // if we are skipping > (rigWindowGrow * 3) the create separate ranges
    // this is not happening if this rig and previous are in the same "interval", this means
    // we explicitly want to keep all the frames between them.
    if (!noSeparateRanges && !rigIndices.empty() &&
        ((rigIndices.back() < rigIndex - rigWindowGrow * 3) ||
         anyBreakInRange(rigIndices.back(), rigIndex))) {
      ranges.back().rigIndexEnd = growUp(rigIndices.back(), rigWindowGrow) + 1;
      ranges.push_back({.rigIndexStart = growDown(rigIndex, rigWindowGrow), .rigIndexEnd = -1});
    }

    rigIndices.push_back(rigIndex);
    rigIndex_to_krIndex[rigIndex] = krIndex;
    prevIntervalStart = newIntervalStart;
  }

  // set beginning and end
  ranges[0].rigIndexStart = growDown(rigIndices[0], rigWindowGrow);
  ranges.back().rigIndexEnd = growUp(rigIndices.back(), rigWindowGrow) + 1;

  return {
      .rigIndices = std::move(rigIndices),
      .rigIndex_to_krIndex = std::move(rigIndex_to_krIndex),
      .ranges = std::move(ranges),
  };
}

void SingleSessionAdapter::initRigs(int64_t rigIndexStart, int64_t rigIndexEnd) {
  for (size_t i = rigIndexStart; i < rigIndexEnd; i++) {
    const auto& iPose = fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[i]];
    prob_.inertialPose_set(
        i, iPose.T_w_IMU.inverse(), iPose.v_w, iPose.omega_bodyImu, iPose.timestamp_us);
  }
}

void SingleSessionAdapter::initRigsFromGtTrajectory(
    int64_t rigIndexStart,
    int64_t rigIndexEnd,
    const TrajectoryBase* trajectory,
    bool poseToGt,
    bool velToGt,
    bool omegaToGt) {
  initRigsFromGtTrajectory(
      {{.rigIndexStart = rigIndexStart, .rigIndexEnd = rigIndexEnd}},
      trajectory,
      poseToGt,
      velToGt,
      omegaToGt);
}

void SingleSessionAdapter::initRigsFromGtTrajectory(
    const std::vector<TimeStampInterval>& timeIntervals,
    const std::vector<int64_t>& krTimestampsUs,
    const TrajectoryBase* trajectory,
    bool poseToGt,
    bool velToGt,
    bool omegaToGt,
    int rigWindowGrow) {
  auto iRef = computeKeyRigInitRef(timeIntervals, krTimestampsUs, rigWindowGrow);
  initRigsFromGtTrajectory(iRef.ranges, trajectory, poseToGt, velToGt, omegaToGt);
}

void SingleSessionAdapter::initRigsFromGtTrajectory(
    const std::vector<Range>& ranges,
    const TrajectoryBase* trajectory,
    bool poseToGt,
    bool velToGt,
    bool omegaToGt) {
  XR_CHECK(trajectory);
  bool anyVel = velToGt || omegaToGt;
  XR_CHECK(poseToGt || anyVel); // at least one should be set
  XR_CHECK(!anyVel || trajectory->haveVelocities());

  for (const auto& range : ranges) {
    for (size_t i = range.rigIndexStart; i < range.rigIndexEnd; i++) {
      const auto& iPose = fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[i]];

      // we need to transform linear velocities to the coordinate system used for poses
      if (!anyVel) {
        SE3 T_bodyImu_world = trajectory->T_bodyImu_world(iPose.timestamp_us);
        prob_.inertialPose_set(
            i,
            T_bodyImu_world,
            T_bodyImu_world.so3().inverse() * iPose.T_w_IMU.so3().inverse() * iPose.v_w,
            iPose.omega_bodyImu,
            iPose.timestamp_us);
      } else {
        const auto ip = trajectory->inertialPose(iPose.timestamp_us);
        const SE3 T_bodyImu_world = poseToGt ? ip.T_bodyImu_world : iPose.T_w_IMU.inverse();
        const Vec3 vel_bodyImu = velToGt ? ip.T_bodyImu_world.so3() * ip.vel_world
                                         : iPose.T_w_IMU.so3().inverse() * iPose.v_w;
        const Vec3 omega = omegaToGt ? ip.omega : iPose.omega_bodyImu;
        prob_.inertialPose_set(
            i,
            T_bodyImu_world,
            T_bodyImu_world.so3().inverse() * vel_bodyImu,
            omega,
            iPose.timestamp_us);
      }
    }
  }
}

double SingleSessionAdapter::walkedDistance(int startFrame, int finalFrame) {
  double totDistance = 0;
  for (int i = startFrame; i < finalFrame; i++) {
    const auto& prevIPose = fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[i]];
    const auto& nextIPose = fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[i + 1]];
    const Vec3 t_prevImu_nextImu =
        prevIPose.T_w_IMU.translation() - nextIPose.T_w_IMU.translation();
    totDistance += t_prevImu_nextImu.norm();
  }
  return totDistance;
}

// (very generous) relative distortion when adapting per-session rel poses to map's rel poses
static constexpr double kWalkedDistanceAdd = 0.5; // add this to walked distance between key-rigs
static constexpr double kRotDriftDegsPerSqrtMeter = 0.2;
static constexpr double kTrDriftCmPerSqrtMeter = 2.0;
static constexpr int64_t kTimeDeltaCloseToResetUs = 300'000; // 0.3 sec

/*
  This function loads a trajectory which had been incroporated into a map,
  and have poses provided in `Ts_bodyImu_world`. Since only a subset of recording's frames
  will be present in the map, the loaded poses will contain neighboring frames, and
  the inertial poses of these frames will be adapted to the transformation traj->map.
  Since this transformation is not constant, an interpolation will be used.
  Velocities are also transformed accordingly, and rescaled if a rescaling occurred
  between trajectory and map.
*/
void SingleSessionAdapter::initRigsInterpolatingPoses(
    const std::vector<TimeStampInterval>& timeIntervals,
    const std::vector<int64_t>& krTimestampsUs,
    const std::vector<SE3>& Ts_bodyImu_world,
    const TrajectoryBase* trajectory,
    bool velToGt,
    bool omegaToGt,
    StatsValueContainer* frameDistortionStats_relRot,
    StatsValueContainer* frameDistortionStats_relTr,
    int rigWindowGrow) {
  XR_CHECK_EQ(krTimestampsUs.size(), Ts_bodyImu_world.size());

  auto iRef = computeKeyRigInitRef(timeIntervals, krTimestampsUs, rigWindowGrow);

  // collect some stats on the "distortion" occurring adapting traj poses to map poses
  StatsValueContainer localFrameDistortionStats_relRot, localFrameDistortionStats_relTr;
  for (size_t nextKrIndex = 1; nextKrIndex < Ts_bodyImu_world.size(); nextKrIndex++) {
    const int64_t prevKrIndex = nextKrIndex - 1;
    const int64_t prevRigIndex =
        findOrDie(matcher_.timestampToRigIndex, krTimestampsUs[prevKrIndex]);
    const int64_t nextRigIndex =
        findOrDie(matcher_.timestampToRigIndex, krTimestampsUs[nextKrIndex]);

    // skip if we have a reset in the middle of the interval
    bool closeToReset = false;
    const auto resetIt = matcher_.resetRigIndices.lower_bound(prevRigIndex);
    if (resetIt != matcher_.resetRigIndices.end()) {
      const int64_t resetIndex = *resetIt;
      const auto& resetIPose =
          fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[resetIndex]];
      if (resetIPose.timestamp_us <= krTimestampsUs[nextKrIndex] + kTimeDeltaCloseToResetUs) {
        closeToReset = true;
      }
    }

    const auto& prevIPose =
        fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[prevRigIndex]];
    const auto& nextIPose =
        fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[nextRigIndex]];
    if (closeToReset) {
      XR_LOGI(
          "Rigs [{}@{}..{}@{}]: Skipping stat because of reset",
          prevRigIndex,
          prevIPose.timestamp_us,
          nextRigIndex,
          nextIPose.timestamp_us);
      continue;
    }

    const SE3 map_T_prev_next =
        Ts_bodyImu_world[prevKrIndex] * Ts_bodyImu_world[nextKrIndex].inverse();
    const SE3 traj_T_prev_next = prevIPose.T_w_IMU.inverse() * nextIPose.T_w_IMU;
    const SE3 distAtPrev = traj_T_prev_next * map_T_prev_next.inverse();
    const double distortionRotDeg = radiansToDegrees(distAtPrev.so3().log().norm());
    const double distortionTrCm = distAtPrev.translation().norm() * 100.0;
    const double walkedDist = walkedDistance(prevRigIndex, nextRigIndex);
    const double sqrtWalkedDist = std::sqrt(walkedDist + kWalkedDistanceAdd);
    const double frameDistortion_relRot =
        distortionRotDeg / (kRotDriftDegsPerSqrtMeter * sqrtWalkedDist);
    const double frameDistortion_relTr = distortionTrCm / (kTrDriftCmPerSqrtMeter * sqrtWalkedDist);

    // collect stats locally, to print them later
    if (verbosity_ != Muted) {
      localFrameDistortionStats_relRot.add(frameDistortion_relRot);
      localFrameDistortionStats_relTr.add(frameDistortion_relTr);
    }

    if (frameDistortionStats_relRot || frameDistortionStats_relTr) {
      if (frameDistortionStats_relRot) {
        frameDistortionStats_relRot->add(frameDistortion_relRot);
      }
      if (frameDistortionStats_relTr) {
        frameDistortionStats_relTr->add(frameDistortion_relTr);
      }
    }
  }

  if (verbosity_ != Muted) {
    XR_LOGI(
        "Adapting {} trajectory poses to map:\n"
        "  distortion rot (rel to expected {:.02f} deg/sqrt(m))   p50: {:.02f}, p90: {:.02f}, p97: {:.02f}, max: {:.02f}\n"
        "  distortion transl (rel to expected {:.02f} cm/sqrt(m)) p50: {:.02f}, p90: {:.02f}, p97: {:.02f}, max: {:.02f}",
        Ts_bodyImu_world.size(),
        kRotDriftDegsPerSqrtMeter,
        localFrameDistortionStats_relRot.p50(),
        localFrameDistortionStats_relRot.p90(),
        localFrameDistortionStats_relRot.pX(97),
        localFrameDistortionStats_relRot.max(),
        kTrDriftCmPerSqrtMeter,
        localFrameDistortionStats_relTr.p50(),
        localFrameDistortionStats_relTr.p90(),
        localFrameDistortionStats_relTr.pX(97),
        localFrameDistortionStats_relTr.max());
  }

  const auto logScalings = computeLogScalings(krTimestampsUs, Ts_bodyImu_world, iRef.rigIndices);

  for (const auto& range : iRef.ranges) {
    for (size_t rigIndex = range.rigIndexStart; rigIndex < range.rigIndexEnd; rigIndex++) {
      const auto& iPose = fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[rigIndex]];
      const auto it_eqOrBigger = iRef.rigIndex_to_krIndex.lower_bound(rigIndex);

      //  Estimated relative scaling between traj and provided map poses is used to:
      // 1. rescale velocities accordingly, if a rescaling occurs we should re-estimate velocities
      // 2. rescale the segments before-first or after-last matched frames
      const double scalingFactor_kr_to_traj = scalingAtTimestamp(logScalings, iPose.timestamp_us);

      Sophus::SE3d T_trajWorld_krWorld;

      // == holds?
      if (it_eqOrBigger != iRef.rigIndex_to_krIndex.end() && it_eqOrBigger->first == rigIndex) {
        const int krIndex = it_eqOrBigger->second;
        T_trajWorld_krWorld = iPose.T_w_IMU * Ts_bodyImu_world[krIndex];
      } else if (it_eqOrBigger == iRef.rigIndex_to_krIndex.begin()) { // before first matched rig?
        const auto [firstMatchedRigIndex, firstMatchedKrIndex] = *it_eqOrBigger;
        const auto& firstMatchIPose =
            fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[firstMatchedRigIndex]];
        T_trajWorld_krWorld = firstMatchIPose.T_w_IMU * Ts_bodyImu_world[firstMatchedKrIndex];
      } else { // after last matched rig?
        const auto it_strictlySmaller = std::prev(it_eqOrBigger);
        if (it_eqOrBigger == iRef.rigIndex_to_krIndex.end()) {
          const auto [lastMatchedRigIndex, lastMatchedKrIndex] = *it_strictlySmaller;
          const auto& lastMatchIPose =
              fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[lastMatchedRigIndex]];
          T_trajWorld_krWorld = lastMatchIPose.T_w_IMU * Ts_bodyImu_world[lastMatchedKrIndex];
        } else {
          // need to interpolate between matched rig before and after
          const auto [prevMatchedRigIndex, prevMatchedKrIndex] = *it_strictlySmaller;
          const auto [nextMatchedRigIndex, nextMatchedKrIndex] = *it_eqOrBigger;
          const auto& prevMatchIPose =
              fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[prevMatchedRigIndex]];
          const auto& nextMatchIPose =
              fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[nextMatchedRigIndex]];
          const Sophus::SE3d prev_T_trajWorld_krWorld =
              prevMatchIPose.T_w_IMU * Ts_bodyImu_world[prevMatchedKrIndex];
          const Sophus::SE3d next_T_trajWorld_krWorld =
              nextMatchIPose.T_w_IMU * Ts_bodyImu_world[nextMatchedKrIndex];

          // compute timestamp in interval and apply SE3-interpolation
          const int64_t prevKrTime = krTimestampsUs[prevMatchedKrIndex];
          const int64_t nextKrTime = krTimestampsUs[nextMatchedKrIndex];
          const double relativeTime =
              double(iPose.timestamp_us - prevKrTime) / (nextKrTime - prevKrTime);
          T_trajWorld_krWorld =
              Sophus::interpolate(prev_T_trajWorld_krWorld, next_T_trajWorld_krWorld, relativeTime);
        }
      }

      SE3 T_bodyImu_world = iPose.T_w_IMU.inverse() * T_trajWorld_krWorld;
      Vec3 vel_traj = iPose.v_w;
      Vec3 omega_imu = iPose.omega_bodyImu;
      if (velToGt || omegaToGt) {
        const auto ip = trajectory->inertialPose(iPose.timestamp_us);
        if (velToGt) {
          vel_traj = iPose.T_w_IMU.so3() * ip.T_bodyImu_world.so3() * ip.vel_world;
        }
        if (omegaToGt) {
          omega_imu = ip.omega;
        }
      }
      Vec3 vel_world = T_trajWorld_krWorld.so3().inverse() * vel_traj * scalingFactor_kr_to_traj;
      prob_.inertialPose_set(rigIndex, T_bodyImu_world, vel_world, omega_imu, iPose.timestamp_us);
    }
  }
}

} // namespace visual_inertial_ba
