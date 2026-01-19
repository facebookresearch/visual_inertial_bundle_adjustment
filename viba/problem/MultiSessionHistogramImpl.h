/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <viba/common/Enumerate.h>
#include <viba/common/Histogram.h>
#include <viba/problem/Histograms.h>
#include <viba/problem/MultiSessionProblem.h>

namespace visual_inertial_ba {

template <typename KeyRigId, typename MapPointId>
void MultiSessionProblem<KeyRigId, MapPointId>::showHistogram(
    bool simpleStats,
    bool separatePerRecordingStats) const {
  auto indexOfRecOwningVar =
      [this](
          bool (SingleSessionProblem::*isOwnVarTypeFunc)(small_thing::VarBase*) const,
          small_thing::VarBase* var) -> int {
    for (const auto& [i, tProb] : enumerate(tProbs_)) {
      if (((*tProb.problem).*isOwnVarTypeFunc)(var)) {
        return i;
      }
    }
    return -1;
  };

  std::vector<std::string> recordingLabels;
  if (separatePerRecordingStats) {
    for (const auto& tProb : tProbs_) {
      recordingLabels.push_back("Recording '" + tProb.label + "''s ");
    }
  } else {
    recordingLabels.emplace_back("Cumulative ");
  }

  Histograms h(opt_);
  if (simpleStats) {
    h.showPixelErrors = false; // image-distance pixel reproj errors
    h.showRotVelPos = false; // separate histograms for rot/vel/pos
    h.separateSecondaryInertial = false; // separate main/secondary imu
    h.showAggregateCalibFactors = true; // one histogram for all rw factors
  }
  h.visual.groupLabel = {"Global (loop closing) "};
  for (const auto& recLab : recordingLabels) {
    h.visual.groupLabel.push_back(recLab + "tracking ");
  }
  int maxGroupIndex = separatePerRecordingStats ? h.visual.groupLabel.size() - 1 : 1;
  h.visual.factorToGroup = [&, maxGroupIndex](
                               small_thing::FactorStoreBase* storeBase, int64_t costIndex) {
    // index of point variable in visual factor arguments: 0 (VisualFactor.cpp)
    // return (index of recording + 1) so 0 (=not found) means global loop closing points
    return std::min(
        indexOfRecOwningVar(
            &SingleSessionProblem::isOwnPointVar, storeBase->costVar(costIndex, 0)) +
            1,
        maxGroupIndex);
  };
  h.visual.groupCol = [](int groupIndex) {
    return groupIndex == 0 ? Histograms::Cyan : Histograms::Green;
  };

  h.inertial.groupLabel = recordingLabels;
  h.secondaryInertial.groupLabel = recordingLabels;
  h.rwImuCalib.groupLabel = recordingLabels;
  h.rwImuExtr.groupLabel = recordingLabels;
  h.rwCamIntr.groupLabel = recordingLabels;
  h.rwCamExtr.groupLabel = recordingLabels;
  h.omegaPriors.groupLabel = recordingLabels;
  h.fpImuCalib.groupLabel = recordingLabels;
  h.fpImuExtr.groupLabel = recordingLabels;
  h.fpCamIntr.groupLabel = recordingLabels;
  h.fpCamExtr.groupLabel = recordingLabels;

  std::unordered_map<const small_thing::VarBase*, int> omegaVarToRec;
  if (separatePerRecordingStats) {
    h.inertial.factorToGroup = [&](small_thing::FactorStoreBase* storeBase, int64_t costIndex) {
      // index of imu calib variable in inertial factor arguments: 0 (InertialFactor.cpp)
      return indexOfRecOwningVar(
          &SingleSessionProblem::isOwnImuCalibVar, storeBase->costVar(costIndex, 0));
    };

    h.secondaryInertial.factorToGroup = [&](small_thing::FactorStoreBase* storeBase,
                                            int64_t costIndex) {
      // index of imu calib variable in inertial factor arguments: 0 (InertialFactor.cpp)
      return indexOfRecOwningVar(
          &SingleSessionProblem::isOwnImuCalibVar, storeBase->costVar(costIndex, 0));
    };

    h.rwImuCalib.factorToGroup = [&](small_thing::FactorStoreBase* storeBase, int64_t costIndex) {
      // index of (one) imu calib variable in imu random walk arguments: 0 (RandomWalkFactors.cpp)
      return indexOfRecOwningVar(
          &SingleSessionProblem::isOwnImuCalibVar, storeBase->costVar(costIndex, 0));
    };

    h.rwCamIntr.factorToGroup = [&](small_thing::FactorStoreBase* storeBase, int64_t costIndex) {
      // index of (one) cam intrinsics variable in imu random walk arguments: 0
      // (RandomWalkFactor.cpp)
      return indexOfRecOwningVar(
          &SingleSessionProblem::isOwnCameraModelVar, storeBase->costVar(costIndex, 0));
    };

    h.rwImuExtr.factorToGroup = [&](small_thing::FactorStoreBase* storeBase, int64_t costIndex) {
      // index of (one) cam extrinsics variable in imu random walk arguments: 0
      // (RandomWalkFactor.cpp)
      return indexOfRecOwningVar(
          &SingleSessionProblem::isOwnImuExtrinsicsVar, storeBase->costVar(costIndex, 0));
    };

    h.rwCamExtr.factorToGroup = [&](small_thing::FactorStoreBase* storeBase, int64_t costIndex) {
      // index of (one) cam extrinsics variable in imu random walk arguments: 0
      // (RandomWalkFactor.cpp)
      return indexOfRecOwningVar(
          &SingleSessionProblem::isOwnCamExtrinsicsVar, storeBase->costVar(costIndex, 0));
    };

    // build map to find the recording index from omega var
    for (const auto& [i, tProb] : enumerate(tProbs_)) {
      for (const auto& [_, rigVar] : tProb.problem->inertialPoses()) {
        omegaVarToRec[&rigVar.omega] = i;
      }
    }
    h.omegaPriors.factorToGroup = [&](small_thing::FactorStoreBase* storeBase, int64_t costIndex) {
      // index of (one) cam extrinsics variable in imu random walk arguments: 0
      // (OmegaPriorFactor.cpp)
      return findOrDie(omegaVarToRec, storeBase->costVar(costIndex, 0));
    };

    h.fpImuCalib.factorToGroup = [&](small_thing::FactorStoreBase* storeBase, int64_t costIndex) {
      // index of (one) imu calib variable in imu random walk arguments: 0 (RandomWalkFactors.cpp)
      return indexOfRecOwningVar(
          &SingleSessionProblem::isOwnImuCalibVar, storeBase->costVar(costIndex, 0));
    };

    h.fpCamIntr.factorToGroup = [&](small_thing::FactorStoreBase* storeBase, int64_t costIndex) {
      // index of (one) cam intrinsics variable in imu random walk arguments: 0
      // (PrioFactor.cpp)
      return indexOfRecOwningVar(
          &SingleSessionProblem::isOwnCameraModelVar, storeBase->costVar(costIndex, 0));
    };

    h.fpImuExtr.factorToGroup = [&](small_thing::FactorStoreBase* storeBase, int64_t costIndex) {
      // index of (one) cam extrinsics variable in imu random walk arguments: 0
      // (PrioFactor.cpp)
      return indexOfRecOwningVar(
          &SingleSessionProblem::isOwnImuExtrinsicsVar, storeBase->costVar(costIndex, 0));
    };

    h.fpCamExtr.factorToGroup = [&](small_thing::FactorStoreBase* storeBase, int64_t costIndex) {
      // index of (one) cam extrinsics variable in imu random walk arguments: 0
      // (PrioFactor.cpp)
      return indexOfRecOwningVar(
          &SingleSessionProblem::isOwnCamExtrinsicsVar, storeBase->costVar(costIndex, 0));
    };
  }

  h.show();
};

} // namespace visual_inertial_ba
