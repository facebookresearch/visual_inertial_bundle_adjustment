/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-ParameterUncheckedArrayBounds
#include <viba/common/Enumerate.h>
#include <viba/single_session/SingleSessionAdapter.h>

#define DEFAULT_LOG_CHANNEL "ViBa::InitRescaling"
#include <logging/Log.h>

namespace visual_inertial_ba {

namespace {

constexpr double minSqRadiusForComparisonM = 4.0 * 4.0; // 4 meters
constexpr double maxSqRadiusForRestartM = 2.0 * 2.0; // 2 meter

// find ranges where for evaluation of relative scaling, having minimum radius
std::vector<std::pair<int64_t, int64_t>> scalingEvalRanges(
    const std::vector<SE3>& Ts_kr_world,
    Verbosity verbosity) {
  std::vector<std::pair<int64_t, int64_t>> ranges;
  int64_t rangeStart = 0, rangeEnd = 1;

  if (verbosity != Muted) {
    XR_LOGI("Computing ranges...");
  }
  while (true) {
    // find end as first rig at distance bigger than scaling comp
    while (rangeEnd < Ts_kr_world.size()) {
      SE3 T_krEnd_krStart = Ts_kr_world[rangeEnd] * Ts_kr_world[rangeStart].inverse();
      double sqDist_krEnd_krStart = T_krEnd_krStart.translation().squaredNorm();
      if (sqDist_krEnd_krStart > minSqRadiusForComparisonM) {
        break;
      }
      rangeEnd++;
    }
    if (rangeEnd >= Ts_kr_world.size()) {
      break;
    }
    ranges.emplace_back(rangeStart, rangeEnd);

    while (rangeStart < rangeEnd) {
      rangeStart++;
      SE3 T_krEnd_krStart = Ts_kr_world[rangeEnd] * Ts_kr_world[rangeStart].inverse();
      double sqDist_krEnd_krStart = T_krEnd_krStart.translation().squaredNorm();
      if (sqDist_krEnd_krStart < maxSqRadiusForRestartM) {
        break;
      }
    }
  }

  if (ranges.empty()) {
    ranges.emplace_back(0, Ts_kr_world.size() - 1);
  }

  if (verbosity != Muted) {
    XR_LOGI("Computing ranges done");
  }

  return ranges;
}

} // namespace

// get interpolated value, or closest value at extremes
double SingleSessionAdapter::scalingAtTimestamp(
    const std::map<int64_t, double>& logScalings,
    int64_t timestamp) {
  const auto it_eqOrBigger = logScalings.lower_bound(timestamp);
  double logScaling;
  if (it_eqOrBigger == logScalings.begin()) {
    logScaling = it_eqOrBigger->second;
  } else {
    const auto it_strictlySmaller = std::prev(it_eqOrBigger);
    if (it_eqOrBigger == logScalings.end()) {
      logScaling = it_strictlySmaller->second;
    } else {
      logScaling = it_strictlySmaller->second +
          (it_eqOrBigger->second - it_strictlySmaller->second) *
              double(timestamp - it_strictlySmaller->first) /
              (it_eqOrBigger->first - it_strictlySmaller->first);
    }
  }

  return exp(logScaling);
}

// estimate relative scaling at selected frame ranges
std::map<int64_t, double> SingleSessionAdapter::computeLogScalings(
    const std::vector<int64_t>& krTimestampsUs,
    const std::vector<SE3>& Ts_kr_world,
    std::vector<int> rigIndices) {
  const auto ranges = scalingEvalRanges(Ts_kr_world, verbosity_);

  std::map<int64_t, double> logScalings;
  for (auto [rangeStart, rangeEnd] : ranges) {
    const int64_t midTimestampUs = (krTimestampsUs[rangeEnd] + krTimestampsUs[rangeStart]) / 2;
    if (rangeStart == rangeEnd) {
      logScalings[midTimestampUs] = 0;
      continue;
    }

    const SE3 T_krEnd_krStart = Ts_kr_world[rangeEnd] * Ts_kr_world[rangeStart].inverse();
    const double sqDist_krEnd_krStart = T_krEnd_krStart.translation().squaredNorm();

    const auto& iPssStart =
        fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[rigIndices[rangeStart]]];
    const auto& iPssEnd =
        fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[rigIndices[rangeEnd]]];
    const SE3 T_rigEnd_rigStart = iPssEnd.T_w_IMU.inverse() * iPssStart.T_w_IMU;
    const double sqDist_rigEnd_rigStart = T_rigEnd_rigStart.translation().squaredNorm();

    const double logScaling = log(sqDist_krEnd_krStart / sqDist_rigEnd_rigStart) * 0.5;
    logScalings[midTimestampUs] = logScaling;
  }

  return logScalings;
}

} // namespace visual_inertial_ba
