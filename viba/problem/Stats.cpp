/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
#include <viba/common/StatsValueContainer.h>
#include <viba/problem/Stats.h>

namespace visual_inertial_ba {

void collectStats(small_thing::Optimizer& opt, const StatsRef& ref) {
  std::vector<small_thing::FactorStoreBase*> inertialFactors;
  small_thing::FactorStoreBase* imuCalibRWFactors = nullptr;
  small_thing::FactorStoreBase* camIntrRWFactors = nullptr;
  small_thing::FactorStoreBase* imuExtrRWFactors = nullptr;
  small_thing::FactorStoreBase* camExtrRWFactors = nullptr;
  small_thing::FactorStoreBase* imuCalibFCFactors = nullptr;
  small_thing::FactorStoreBase* camIntrFCFactors = nullptr;
  small_thing::FactorStoreBase* imuExtrFCFactors = nullptr;
  small_thing::FactorStoreBase* camExtrFCFactors = nullptr;

  // `opt.factorStores` are opaque classes (`FactorStoreBase`) containing an instance of
  // `FactorStore<...>`, which can give us a prettified `typeid(F).name()` where F is the function
  // type specified in `addFactor(F&& f,...)`; those `factorStores` group together the factors of
  // the same type. Our costs are lambdas, and they contain the enclosing function in their
  // prettified type name, so we search string bits to get the relevant bucket of cost terms.
  for (const auto& [tid, factorStore] : opt.factorStores.stores) {
    std::string name = factorStore->name();

    if (name.find("visual_inertial_ba::InertialFactor") != std::string::npos ||
        name.find("visual_inertial_ba::SecondaryImuInertialFactor") != std::string::npos) {
      inertialFactors.push_back(factorStore.get());
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addImuCalibRWFactor(") !=
        std::string::npos) {
      imuCalibRWFactors = factorStore.get();
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addCamIntrRWFactor(") !=
        std::string::npos) {
      camIntrRWFactors = factorStore.get();
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addImuExtrRWFactor(") !=
        std::string::npos) {
      imuExtrRWFactors = factorStore.get();
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addCamExtrRWFactor(") !=
        std::string::npos) {
      camExtrRWFactors = factorStore.get();
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addImuPrior(") != std::string::npos) {
      imuCalibFCFactors = factorStore.get();
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addCamIntrinsicsPrior(") !=
        std::string::npos) {
      camIntrFCFactors = factorStore.get();
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addCamExtrinsicsPrior(") !=
        std::string::npos) {
      imuExtrFCFactors = factorStore.get();
    } else if (
        name.find("visual_inertial_ba::SingleSessionProblem::addImuExtrinsicsPrior(") !=
        std::string::npos) {
      camExtrFCFactors = factorStore.get();
    }
  }

  if (ref.rotErrDeg || ref.velErrCmS || ref.posErrCm || ref.imu) {
    for (auto* factorStore : inertialFactors) {
      for (int64_t count = factorStore->numCosts(), i = 0; i < count; i++) {
        if (ref.rotErrDeg || ref.velErrCmS || ref.posErrCm) {
          Vec9 rotVelPosE;
          factorStore->unweightedError(i, rotVelPosE);
          if (ref.rotErrDeg) {
            const double rotErrDegrees = radiansToDegrees(rotVelPosE.head<3>().norm());
            ref.rotErrDeg->add(rotErrDegrees);
          }
          if (ref.velErrCmS) {
            const double velErrCmS = rotVelPosE.segment<3>(3).norm() * 100.0;
            ref.velErrCmS->add(velErrCmS);
          }
          if (ref.posErrCm) {
            const double posErrCm = rotVelPosE.tail<3>().norm() * 100.0;
            ref.posErrCm->add(posErrCm);
          }
        }
        if (ref.imu) {
          const double totalInertialError =
              std::sqrt(factorStore->covWeightedSquaredError(i) * 2.0);
          ref.imu->add(totalInertialError);
        }
      }
    }
  }

  auto addCalibError = [&](small_thing::FactorStoreBase* factorStore,
                           StatsValueContainer* sv1,
                           StatsValueContainer* sv2) {
    if (!factorStore || !sv1 || !sv2) {
      return;
    }
    for (int64_t count = factorStore->numCosts(), i = 0; i < count; i++) {
      const double error = std::sqrt(factorStore->covWeightedSquaredError(i) * 2.0);
      if (sv1) {
        sv1->add(error);
      }
      if (sv2) {
        sv2->add(error);
      }
    }
  };
  addCalibError(imuCalibRWFactors, ref.imuCalibRw, ref.anyRw);
  addCalibError(camIntrRWFactors, ref.camIntrRw, ref.anyRw);
  addCalibError(imuExtrRWFactors, ref.imuExtrRw, ref.anyRw);
  addCalibError(camExtrRWFactors, ref.camExtrRw, ref.anyRw);
  addCalibError(imuCalibFCFactors, ref.imuCalibPrio, ref.anyPrio);
  addCalibError(camIntrFCFactors, ref.camIntrPrio, ref.anyPrio);
  addCalibError(imuExtrFCFactors, ref.imuExtrPrio, ref.anyPrio);
  addCalibError(camExtrFCFactors, ref.camExtrPrio, ref.anyPrio);
}

} // namespace visual_inertial_ba
