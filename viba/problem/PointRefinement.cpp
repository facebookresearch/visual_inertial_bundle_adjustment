/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-ParameterUncheckedArrayBounds
#include <viba/problem/PointRefinement.h>
#include <viba/problem/Types.h>

#include <logging/Checks.h>
#define DEFAULT_LOG_CHANNEL "ViBa::PointRefinement"
#include <logging/Log.h>

namespace visual_inertial_ba {

using PointVariable = small_thing::Variable<Vec3>;

using FactorList = std::vector<std::pair<small_thing::FactorStoreBase*, int64_t>>;

static std::unordered_map<PointVariable*, FactorList> perPointTracks(small_thing::Optimizer& opt) {
  std::vector<small_thing::FactorStoreBase*> allFactors;

  // use introspection to retrieve the factors and variables
  for (const auto& [tid, factorStore] : opt.factorStores.stores) {
    std::string name = factorStore->name();
    if (name.find("visual_inertial_ba::VisualFactor") != std::string::npos ||
        name.find("visual_inertial_ba::RollingShutterVisualFactor") != std::string::npos ||
        name.find("visual_inertial_ba::BaseMapVisualFactor") != std::string::npos) {
      allFactors.push_back(factorStore.get());
    }
  }

  std::unordered_map<PointVariable*, FactorList> ptVarToFactorList;
  for (auto* factors : allFactors) {
    for (int64_t count = factors->numCosts(), i = 0; i < count; i++) {
      PointVariable* ptVar = static_cast<PointVariable*>(
          factors->costVar(i, 0)); // 0-th in both Visual/BaseMap factors
      auto& factorList = ptVarToFactorList[ptVar];
      factorList.emplace_back(factors, i);
    }
  }

  return ptVarToFactorList;
}

// compute gradient/hessian - also update backup cost values
static std::tuple<double, Vec3, Mat33>
pointGradHess(std::vector<double>& costBackups, const FactorList& factorList, bool updateBackups) {
  Vec3 accumGrad = Vec3::Zero();
  Mat33 accumHess = Mat33::Zero();
  double accumCost = 0.0;
  costBackups.resize(factorList.size());
  for (size_t q = 0; q < factorList.size(); q++) {
    const auto& [f, k] = factorList[q];
    Vec3 grad;
    Mat33 hess;
    auto maybeCost = f->varGradHess(k, 0, grad, hess);
    if (maybeCost.has_value()) {
      accumCost += *maybeCost;
      accumGrad += grad;
      accumHess += hess;
      if (updateBackups) {
        costBackups[q] = *maybeCost;
      }
    } else if (updateBackups) {
      costBackups[q] = -1.0;
    }
  }
  return {accumCost, accumGrad, accumHess};
};

// compute cost - for failing observations return backup value
static double pointCost(const std::vector<double>& costBackups, const FactorList& factorList) {
  double newAccumCost = 0.0;
  for (size_t q = 0; q < factorList.size(); q++) {
    if (costBackups[q] < 0) {
      continue;
    }
    const auto& [f, k] = factorList[q];
    auto maybeCost = f->singleCost(k);
    newAccumCost += maybeCost.value_or(costBackups[q]);
  }
  return newAccumCost;
};

static constexpr double kLambda = 1e-5;
static constexpr int kMaxNumIterations = 5;
static constexpr double kCostTolerance = 1e-8;
static constexpr double kStepTolerance = 1e-6;
static constexpr double kMinImprovementRatio = 0.2;
static constexpr double kStepReduction = 0.3;

// return: (old cost, new cost, number of iterations)
static std::tuple<double, double, int> optimizeOnePoint(
    PointVariable& ptVar,
    const FactorList& factorList,
    std::vector<double>& costBackups) {
  double startCost = 0.0, endCost = 0.0;
  int nIts = 0;
  for (int i = 0; i < kMaxNumIterations; i++) {
    auto [cost, grad, hess] = pointGradHess(costBackups, factorList, true);
    if (i == 0) {
      startCost = endCost = cost;
    }
    hess.diagonal() *= 1.0 + kLambda;
    hess.diagonal().array() += kLambda;
    Vec3 step = -hess.ldlt().solve(grad);

    Vec3 ptVarBackup = ptVar.value;

    bool success = false;
    ptVar.value += step;
    double newCost = pointCost(costBackups, factorList);
    double modelCostDelta = step.dot(grad);
    if (-modelCostDelta < kCostTolerance) {
      break; // nothing to do
    }
    if (newCost < cost + modelCostDelta * kMinImprovementRatio) {
      success = true;
    } else { // try a reduced step
      auto [nCost, nGrad, nHess] = pointGradHess(costBackups, factorList, false);
      ptVar.value = ptVarBackup;

      // compute size of reduced step
      double newDelta = step.dot(nGrad);
      if (newDelta > 0) {
        step *= -modelCostDelta / (newDelta - modelCostDelta);
      } else {
        step *= kStepReduction;
      }

      // try reduced step
      ptVar.value += step;
      newCost = pointCost(costBackups, factorList);
      if (newCost < cost + step.dot(grad) * kMinImprovementRatio) {
        success = true;
      } else {
        ptVar.value = ptVarBackup;
      }
    }

    if (success) {
      nIts++;
      endCost = newCost;
    } else {
      nIts = -1;
      break;
    }

    if (step.squaredNorm() < kStepTolerance * kStepTolerance) {
      break;
    }
  }

  return {startCost, endCost, nIts};
}

void refinePoints(small_thing::Optimizer& opt, bool muted) {
  std::unordered_map<PointVariable*, FactorList> ptVarToFactorList = perPointTracks(opt);
  if (!muted) {
    XR_LOGI("Refining {} points", ptVarToFactorList.size());
  }

  std::vector<double> costBackups;
  double totalStartCost = 0.0, totalEndCost = 0.0;
  int64_t nFailures = 0;
  int64_t nTotalIterations = 0;
  int64_t nAtLeastOneIt = 0;
  for (auto& [pPtVar, factorList] : ptVarToFactorList) {
    auto [startCost, endCost, nIts] = optimizeOnePoint(*pPtVar, factorList, costBackups);
    if (nIts < 0) {
      nFailures++;
    } else {
      nTotalIterations += nIts;
      nAtLeastOneIt += (nIts > 0);
    }
    totalStartCost += startCost;
    totalEndCost += endCost;
  }

  if (!muted) {
    XR_LOGI(
        "Point refinement:\n"
        "  total cost {:.3e} -> {:.3e} ({})\n"
        "  pts: {}, successfulIts: {}, nFailures: {}, nAtLeastOneIt: {}",
        totalStartCost,
        totalEndCost,
        small_thing::percentageString((totalEndCost - totalStartCost) / totalStartCost),
        ptVarToFactorList.size(),
        nTotalIterations,
        nFailures,
        nAtLeastOneIt);
  }
}

} // namespace visual_inertial_ba
