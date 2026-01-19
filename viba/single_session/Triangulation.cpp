/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
#include <viba/common/Enumerate.h>
#include <viba/single_session/SingleSessionAdapter.h>
#include <viba/single_session/Triangulation.h>
#include <optional>
#include <random>

#define DEFAULT_LOG_CHANNEL "ViBa::Triangulation"
#include <logging/Log.h>

namespace visual_inertial_ba {

using namespace triangulation;

namespace {
// ray (half straight line, from a starting point)
struct Ray {
  Vec3 start;
  Vec3 direction;
};
} // namespace

// search a triangulation candidate from rays. It is a ransac on pairs of rays,
// computing the closest point to the two rays (not optimal because it doesn't minimize
// reprojection errors, but a good enough approximation). The estimate coming from
// pair of rays minimizing a sum of (clamped) angles is selected.
static std::optional<std::pair<Vec3, int>> findTriangulationCandidate(
    const std::vector<Ray>& rays,
    int seed) {
  std::mt19937 mt(seed);
  std::uniform_int_distribution<> aDist(0, rays.size() - 1);
  std::uniform_int_distribution<> offsetDist(1, rays.size() - 1);

  Vec3 bestCandidate;
  double bestAngleSum = std::numeric_limits<double>::infinity();
  int bestNumInliers = 0;
  [[maybe_unused]] int successfulRansacAttempts = 0;

  for (int i = 0; i < kNumRansac; i++) {
    int a = aDist(mt);
    int b = (a + offsetDist(mt)) % rays.size();

    Vec3 ortho = rays[a].direction.cross(rays[b].direction);
    double orthoNorm = ortho.norm();
    if (orthoNorm < 1e-4) {
      continue;
    }
    Vec3 orthoNormalized = ortho / orthoNorm;
    Vec3 aLateral = orthoNormalized.cross(rays[a].direction);
    Vec3 bLateral = orthoNormalized.cross(rays[b].direction);
    double aLateralDotB = aLateral.dot(rays[b].direction);
    double bLateralDotA = bLateral.dot(rays[a].direction);
    double bFact = aLateral.dot(rays[a].start - rays[b].start) / aLateralDotB;
    double aFact = bLateral.dot(rays[b].start - rays[a].start) / bLateralDotA;
    if (bFact < 0.0 || aFact < 0.0) {
      continue;
    }
    Vec3 candidate = rays[a].start + aFact * rays[a].direction +
        orthoNormalized * (0.5 * orthoNormalized.dot(rays[b].start - rays[a].start));

    double angleSum = 0.0;
    int numInliers = 0;
    for (size_t j = 0; j < rays.size(); j++) {
      Vec3 altDir = (candidate - rays[j].start).normalized();
      double angle = 2.0 * asin((rays[j].direction - altDir).norm() * 0.5);
      if (angle < kOutlierObservationRads) {
        angleSum += angle;
        numInliers++;
      } else {
        angleSum += kOutlierObservationRads;
      }
    }

    if (numInliers < kMinNumInliersInTriangulation) {
      continue;
    }

    successfulRansacAttempts++;
    if (angleSum < bestAngleSum) {
      bestNumInliers = numInliers;
      bestCandidate = candidate;
      bestAngleSum = angleSum;
    }
  }
  if (bestNumInliers < kMinNumInliersInTriangulation) {
    return std::nullopt;
  }
  return std::make_pair(bestCandidate, bestNumInliers);
}

template <typename LossType>
void SingleSessionAdapter::refineTriangulationResult(
    TriangulationResult& result,
    const std::vector<int64_t>& obsIndices,
    double outlierThreshold,
    bool skipOutliers,
    int maxNumIterations,
    const LossType& loss) {
  double outlierSquaredThreshold = outlierThreshold * outlierThreshold;

  for (int itNum = 0; itNum < maxNumIterations; itNum++) {
    Vec3 grad = Vec3::Zero();
    Mat33 H = Mat33::Zero();
    double cost = 0.0;

    result.numInlierObservations = 0;
    result.inlierObservationIndices.clear();

    for (const auto& [i, idx] : enumerate(obsIndices)) {
      const auto& procObs = fData_.trackingObservations[idx];
      const int64_t rigIndex = findOrDie(matcher_.timestampToRigIndex, procObs.captureTimestampUs);
      const int64_t cameraIndex = procObs.cameraIndex;

      // get rig and extrinsic variables
      const SE3 T_bodyImu_world = kModelRollingShutter
          ? prob_.T_bodyImu_world_atImageRow(rigIndex, cameraIndex, procObs.projectionBaseRes[1])
          : prob_.inertialPose(rigIndex).T_bodyImu_world.value;
      const auto& T_Cam_BodyImu_var = prob_.T_Cam_BodyImu(rigIndex, cameraIndex).var;

      const SE3 T_cam_world = T_Cam_BodyImu_var.value * T_bodyImu_world;
      const Vec3 camPt = T_cam_world * result.point;

      // unproject to get vector in camera's frame of reference
      Mat23 dImgPt_dCamPt;
      Vec2 imgPt;
      if (!prob_.cameraModel(rigIndex, cameraIndex)
               .var.value.project(camPt, imgPt, dImgPt_dCamPt)) {
        continue;
      }

      const Vec2 imageError = imgPt - procObs.projectionBaseRes.cast<double>();
      const Vec2 weightedError = procObs.sqrtH_BaseRes.cast<double>() * imageError;
      const double squaredImageError = imageError.squaredNorm();

      if (squaredImageError < outlierSquaredThreshold) {
        result.numInlierObservations++;
        result.inlierObservationIndices.push_back(idx);
      } else if (skipOutliers) {
        continue;
      }

      const Mat23 dErr_dWorldPt =
          procObs.sqrtH_BaseRes.cast<double>() * dImgPt_dCamPt * T_cam_world.rotationMatrix();

      const auto [softErr, dSoftErr] = loss.jet2(weightedError.squaredNorm());
      const Mat23 dLoss_dErr_dWorldPt = dSoftErr * dErr_dWorldPt;
      cost += softErr * 0.5;
      grad += weightedError.transpose() * dLoss_dErr_dWorldPt;
      H += dErr_dWorldPt.transpose() * dLoss_dErr_dWorldPt;
    }

    result.point -= H.llt().solve(grad);
  }
}

// triangulation settings
static const auto kRefine1_loss = small_thing::HuberLoss(kRefine1_loss_radius);
static const auto kRefine2_loss = small_thing::HuberLoss(kRefine2_loss_radius);

// triangulate a point
std::optional<TriangulationResult> SingleSessionAdapter::triangulatePoint(
    const std::vector<int64_t>& obsIndices,
    int seed) {
  // count how many observations were inliers
  if (obsIndices.size() < kMinInlierObs) {
    return std::nullopt;
  }

  // compute rays, according to current estimates
  std::vector<Ray> rays(obsIndices.size());
  for (const auto& [indexInTrack, obsIndex] : enumerate(obsIndices)) {
    const auto& procObs = fData_.trackingObservations[obsIndex];
    const int64_t rigIndex = findOrDie(matcher_.timestampToRigIndex, procObs.captureTimestampUs);
    const int64_t cameraIndex = procObs.cameraIndex;

    // get rig and extrinsic variables
    const SE3 T_bodyImu_world = kModelRollingShutter
        ? prob_.T_bodyImu_world_atImageRow(rigIndex, cameraIndex, procObs.projectionBaseRes[1])
        : prob_.inertialPose(rigIndex).T_bodyImu_world.value;
    const auto& T_Cam_BodyImu_var = prob_.T_Cam_BodyImu(rigIndex, cameraIndex).var;
    const SE3 T_world_cam = (T_Cam_BodyImu_var.value * T_bodyImu_world).inverse();

    const Vec3 vectorInCam =
        prob_.cameraModel(rigIndex, cameraIndex)
            .var.value.unprojectNoChecks(procObs.projectionBaseRes.cast<double>());

    rays[indexInTrack] = {
        .start = T_world_cam.translation(),
        .direction = T_world_cam.so3() * vectorInCam.normalized(),
    };
  }

  // rays are used to find a triangulation candidate
  const auto maybeCandidateInl = findTriangulationCandidate(rays, seed);
  if (!maybeCandidateInl) {
    return std::nullopt;
  }

  TriangulationResult result{
      .point = maybeCandidateInl->first,
      .numInlierObservations = maybeCandidateInl->second,
  };
  result.inlierObservationIndices.reserve(obsIndices.size());

  // refine 1
  refineTriangulationResult(
      result,
      obsIndices,
      kRefine1_outlierThreshold,
      kRefine1_skipOutliers,
      kRefine1_maxNumIterations,
      kRefine1_loss);
  if (result.numInlierObservations < kMinNumInliersAfterRefinement) {
    return std::nullopt;
  }

  // refine 2
  refineTriangulationResult(
      result,
      obsIndices,
      kRefine2_outlierThreshold,
      kRefine2_skipOutliers,
      kRefine2_maxNumIterations,
      kRefine2_loss);
  if (result.numInlierObservations < kMinNumInliersAfterRefinement) {
    return std::nullopt;
  }

  return result;
}

} // namespace visual_inertial_ba
