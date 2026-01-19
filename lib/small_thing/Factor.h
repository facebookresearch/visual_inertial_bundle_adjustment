/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <dispenso/parallel_for.h>
#include <small_thing/AtomicOps.h>
#include <small_thing/Print.h>
#include <small_thing/SoftLoss.h>
#include <small_thing/Variable.h>
#include <Eigen/Geometry>
#include <unordered_set>

namespace small_thing {

struct CostStats {
  int64_t numTotal = 0;
  int64_t numInvalid = 0;
  int64_t numPrevInvalid = 0;

  void operator+=(const CostStats& that) {
    numTotal += that.numTotal;
    numInvalid += that.numInvalid;
    numPrevInvalid += that.numPrevInvalid;
  }
};

// "opaque" (ie non-templated) base class for typed factor classes
class FactorStoreBase {
 public:
  virtual ~FactorStoreBase() {}

  virtual int64_t totalErrorDimensionality() const = 0;

  virtual int64_t numCosts() const = 0;

  virtual bool unweightedError(int64_t, Eigen::Ref<Eigen::VectorXd> err) const = 0;

  virtual std::optional<double> varGradHess(
      int64_t,
      int,
      Eigen::Ref<Eigen::VectorXd> grad,
      Eigen::Ref<Eigen::MatrixXd> hess) const = 0;

  virtual double unweightedSquaredError(int64_t) const = 0;

  virtual double covWeightedSquaredError(int64_t) const = 0;

  virtual std::optional<double> singleCost(int64_t i) const = 0;

  virtual VarBase* costVar(int64_t, int) const = 0;

  virtual std::string name() const = 0;

  virtual bool verifyJacobians(
      double epsilon,
      double relativeTolerance,
      double absoluteTolerance,
      bool stopAtFirstError,
      const LogFunc& log) const = 0;

  virtual std::tuple<double, double, double> singleExpectedDelta(
      int64_t k,
      const double* stepData,
      const BaSpaCho::PermutedCoalescedAccessor& acc) const = 0;

  virtual double computeCost(
      bool makeComparableWithStored,
      CostStats* stats = nullptr,
      dispenso::ThreadPool* threadPool = nullptr) const = 0;

  virtual double computeGradHess(
      double* gradData,
      const BaSpaCho::PermutedCoalescedAccessor& acc,
      double* hessData,
      bool updateCachedResults,
      bool dontRetryFailed,
      dispenso::ThreadPool* threadPool = nullptr) const = 0;

  virtual void registerVariables(
      std::vector<int64_t>& paramSizes,
      TypedStore<VariableStoreBase>& varStores) = 0;

  virtual void registerBlocks(
      std::unordered_set<std::pair<int64_t, int64_t>, pair_hash>& blocks) = 0;
};

struct ResultCache {
  mutable double value;
};

using CostAndStats = std::pair<double, CostStats>;

inline CostAndStats aggregate(const std::vector<CostAndStats>& stats) {
  CostAndStats aggreg = stats[0];
  for (size_t i = 1; i < stats.size(); ++i) {
    aggreg.first += stats[i].first;
    aggreg.second += stats[i].second;
  }
  return aggreg;
}

// specific class storing factors, can be casted down to "opaque" non-templated FactorStoreBase
template <typename Factor, bool HasPrecisionMatrix, typename SoftLoss, typename... Variables>
class FactorStore : public FactorStoreBase {
 public:
  using RawErrorType = decltype(std::declval<Factor>()(
      std::declval<typename Variables::DataType>()...,
      (std::declval<Variables>(), NullRef())...));
  static constexpr bool HasOptionalErrors = UnpackOpt<RawErrorType>::isOpt;
  using ErrorType = typename UnpackOpt<RawErrorType>::BaseType;

  static constexpr int ErrorSize = VarSpec<ErrorType>::TangentDimSpec;
  static_assert(
      ErrorSize != Eigen::Dynamic,
      "Dynamic vectors are not supported for factor's return value");
  static constexpr bool HasSoftLoss = !std::is_same<SoftLoss, TrivialLoss>::value;
  using PrecisionMatrix = Eigen::Matrix<double, ErrorSize, ErrorSize>;
  using VarTuple = std::tuple<Variables*...>;

  // build args used for bound factor, they can be from a
  // base minimum of (Factor, VarTuple) to a max of
  //   (Factor, VarTuple, PrecisionMatrix, const SoftLoss*, mutable double)
  using BaseTuple = std::tuple<Factor, VarTuple>;

  static constexpr int PrecisionMatrixIndex =
      HasPrecisionMatrix ? std::tuple_size_v<BaseTuple> : -1;
  using TupleUpToPrecision = std::conditional_t<
      HasPrecisionMatrix, //
      CompoundTuple_t<BaseTuple, PrecisionMatrix>,
      BaseTuple>;

  static constexpr int SoftLossIndex = //
      HasSoftLoss ? std::tuple_size_v<TupleUpToPrecision> : -1;
  using TupleUpToSoftLoss = std::conditional_t<
      HasSoftLoss, //
      CompoundTuple_t<TupleUpToPrecision, const SoftLoss*>,
      TupleUpToPrecision>;

  static constexpr int OptionalErrorIndex =
      HasOptionalErrors ? std::tuple_size_v<TupleUpToSoftLoss> : -1;
  using TupleUpToOptionalError = std::conditional_t<
      HasOptionalErrors, //
      CompoundTuple_t<TupleUpToSoftLoss, ResultCache>,
      TupleUpToSoftLoss>;

  using TupleType = TupleUpToOptionalError;

  template <typename... Args>
  void addFactor(Args&&... args) {
    if constexpr (HasOptionalErrors) {
      boundFactors.emplace_back(std::forward<Args>(args)..., ResultCache());
    } else {
      boundFactors.emplace_back(std::forward<Args>(args)...);
    }
  }

  // helper, get i-th loss
  auto getLoss(int64_t i) const {
    if constexpr (!HasSoftLoss) {
      return TrivialLoss();
    } else {
      return *std::get<SoftLossIndex>(boundFactors[i]);
    }
  }

  // computes the plain return value of the factor
  RawErrorType rawError(int64_t i) const {
    auto& factor = std::get<0>(boundFactors[i]);
    auto& args = std::get<1>(boundFactors[i]);

    return std::apply( // expanding argument pack...
        [&](auto&&... args) { return factor(args->value..., ((void)args, NullRef())...); },
        args);
  }

  // computes the squared error, possibly applying the covariance
  double squaredError(int64_t i, const ErrorType& err) const {
    if constexpr (HasPrecisionMatrix) {
      return err.dot(std::get<PrecisionMatrixIndex>(boundFactors[i]) * err);
    } else {
      return err.squaredNorm();
    }
  }

  virtual ~FactorStore() override {}

  virtual int64_t totalErrorDimensionality() const override {
    return boundFactors.size() * ErrorSize;
  }

  // [opaque] return number of costs
  virtual int64_t numCosts() const override {
    return boundFactors.size();
  }

  // [opaque] retrieve the error vector, into a Eigen::VectorXd already of the right size
  virtual bool unweightedError(int64_t i, Eigen::Ref<Eigen::VectorXd> err) const override {
    BASPACHO_CHECK_EQ(err.size(), ErrorSize);
    return unpackingOptional(
        rawError(i),
        [&](const auto& e) {
          err = e;
          return false;
        },
        [] { return true; });
  }

  // [opaque] squared error value, not weighted
  virtual double unweightedSquaredError(int64_t i) const override {
    return unpackingOptional(
        rawError(i), //
        [](const auto& e) { return e.squaredNorm() * 0.5; },
        [] { return 0.0; });
  }

  // [opaque] squared error value, covariance weighted
  virtual double covWeightedSquaredError(int64_t i) const override {
    return unpackingOptional(
        rawError(i), //
        [i, this](const auto& e) { return squaredError(i, e) * 0.5; },
        [] { return 0.0; });
  }

  // [opaque] cost as accounted
  virtual std::optional<double> singleCost(int64_t i) const override {
    return computeSingleCost(i, false);
  }

  // (helper) get i-th variable as opaque VarBase
  template <int offset>
  static VarBase* getVar(const VarTuple& tup, int i) {
    if constexpr (offset < sizeof...(Variables)) {
      return (i == offset) ? std::get<offset>(tup) : getVar<offset + 1>(tup, i);
    } else {
      return nullptr;
    }
  }

  // [opaque] get v-th variable of i-th cost, as opaque VarBase
  virtual VarBase* costVar(int64_t i, int v) const override {
    auto& args = std::get<1>(boundFactors[i]);
    return getVar<0>(args, v);
  }

  // [opaque] get a pretty typename of the underlying type
  virtual std::string name() const override {
    return prettyTypeName<FactorStore>();
  }

  // [opaque] do jacobian verification
  virtual bool verifyJacobians(
      double epsilon,
      double relativeTolerance,
      double absoluteTolerance,
      bool stopAtFirstError,
      const LogFunc& log) const override {
    // let's just check a sample
    const int nCheck = std::min(boundFactors.size(), 100UL);

    std::tuple<Eigen::Vector<double, Variables::TangentDimSpec>...> maxRelErr;
    forEach<0, sizeof...(Variables)>([&](auto iWrap) {
      static constexpr int i = decltype(iWrap)::value;
      using iVarType =
          std::remove_reference_t<decltype(std::get<i>(std::declval<std::tuple<Variables...>>()))>;
      if constexpr (iVarType::TangentDimSpec == Eigen::Dynamic) {
        int maxTgDim = 0;
        for (size_t k = 0; k < nCheck; k++) {
          auto& args = std::get<1>(boundFactors[k]);
          iVarType& iVar = *std::get<i>(args);
          maxTgDim = std::max(iVar.getTangentDim(), maxTgDim);
        }
        std::get<i>(maxRelErr).setZero(maxTgDim);
      } else {
        std::get<i>(maxRelErr).setZero();
      }
    });

    bool foundError = false;
    for (size_t k = 0; k < nCheck; k++) {
      auto& factor = std::get<0>(boundFactors[k]);
      auto& args = std::get<1>(boundFactors[k]);

      withStaticRange<0, sizeof...(Variables)>([&](auto... iWrapIndices) {
        std::tuple<Eigen::Map<Eigen::Matrix<double, ErrorSize, Variables::TangentDimSpec>>...>
        jacobians(Eigen::Map<Eigen::Matrix<double, ErrorSize, Variables::TangentDimSpec>>(
            (double*)(alloca(
                sizeof(double) * ErrorSize *
                std::get<decltype(iWrapIndices)::value>(args)->getTangentDim())),
            ErrorSize,
            std::get<decltype(iWrapIndices)::value>(args)->getTangentDim())...);

        // Call factor on args. Compute analytic jacobians as per factor.
        const RawErrorType rawErr = factor(
            std::get<decltype(iWrapIndices)::value>(args)->value...,
            Eigen::Ref<Eigen::Matrix<double, ErrorSize, Variables::TangentDimSpec>>(
                std::get<decltype(iWrapIndices)::value>(jacobians))...);
        if constexpr (HasOptionalErrors) {
          if (!rawErr.has_value()) {
            return;
          }
        }
        const ErrorType& err = unpackedOpt(rawErr);

        // Compute jacobians numerically and check difference with analytic version
        forEach<0, sizeof...(Variables)>([&](auto iWrap) {
          static constexpr int i = decltype(iWrap)::value;
          using iVarType = std::remove_reference_t<decltype(*std::get<i>(args))>;
          iVarType& iVar = *std::get<i>(args);
          using iVarSpec = VarSpec<typename iVarType::DataType>;
          const int iTangentDim = iVar.getTangentDim();
          const int iDataDim = iVar.getDataDim();

          // Cache the original value of the variable as we are going to apply boxPlus operator to
          // it to numerically est jacobian.
          Eigen::Map<Eigen::Vector<double, iVarType::DataDimSpec>> iVarBackup(
              (double*)alloca(sizeof(double) * iDataDim), iDataDim);
          iVarSpec::getData(iVar.value, iVarBackup);
          const auto& iJac = std::get<i>(jacobians);
          Eigen::Map<Eigen::Matrix<double, ErrorSize, iVarType::TangentDimSpec>> iNumJac(
              (double*)alloca(sizeof(double) * ErrorSize * iTangentDim), ErrorSize, iTangentDim);

          Eigen::Map<Eigen::Vector<double, iVarType::TangentDimSpec>> tgStep(
              (double*)alloca(sizeof(double) * iTangentDim), iTangentDim);
          double paramMaxRelErrs = 0;
          int maxRelErrCol = -1;
          for (int t = 0; t < iTangentDim; t++) {
            tgStep.setZero();
            tgStep[t] = epsilon;
            iVarSpec::applyBoxPlus(iVar.value, tgStep);

            RawErrorType pRawErr = std::apply( // expanding argument pack...
                [&](auto&&... args) { return factor(args->value..., ((void)args, NullRef())...); },
                args);
            const ErrorType& pErr = unpackedOpt(pRawErr);
            iNumJac.col(t) = (pErr - err) / epsilon;
            double relErr =
                std::max((iNumJac.col(t) - iJac.col(t)).norm() - absoluteTolerance, 0.0) /
                (iNumJac.col(t).norm() + epsilon);
            std::get<i>(maxRelErr)[t] = std::max(std::get<i>(maxRelErr)[t], relErr);
            if (relErr > paramMaxRelErrs) {
              paramMaxRelErrs = relErr;
              maxRelErrCol = t;
            }

            iVarSpec::setData(iVar.value, iVarBackup); // restore
          }
          if (paramMaxRelErrs > relativeTolerance) {
            if (log && stopAtFirstError) {
              std::stringstream ss;
              ss << "Factor" << k << ".Jac" << i << ":\n"
                 << iJac << "\nwhile numeric Jacobian is\n"
                 << iNumJac << "\n and has relative error " << paramMaxRelErrs << " > "
                 << relativeTolerance << " in column " << maxRelErrCol;
              log(ss.str());
            }
            foundError = true;
          }
        });
      });

      if (foundError && stopAtFirstError) {
        break;
      }
    }

    if (log) {
      std::stringstream ss;
      ss << "Verified factor class:\n"
         << prettyTypeName<FactorStore>() << "\n"
         << "Factors checked: " << nCheck << "/" << boundFactors.size() << ", Jacobian check "
         << (foundError ? "FAILED" : "OK!") << " (relative tolerance: " << relativeTolerance
         << ", absolute tolerance: " << absoluteTolerance << ")\n";
      forEach<0, sizeof...(Variables)>([&](auto iWrap) {
        static constexpr int i = decltype(iWrap)::value;
        ss << "Relative errors in cols of " << i << "-th Jacobian:\n  "
           << std::get<i>(maxRelErr).transpose() << std::endl;
      });
      log(ss.str());
    }

    return !foundError;
  }

  // (helper) compute one cost, applying loss
  std::optional<double>
  computeSingleCost(int64_t i, bool makeComparableWithStored, CostStats* stats = nullptr) const {
    auto rawErr = rawError(i);
    if (stats) {
      stats->numTotal += 1;
    }
    if constexpr (HasOptionalErrors) {
      const double prevValue = std::get<OptionalErrorIndex>(boundFactors[i]).value;
      const bool prevInvalid = prevValue < 0.0;
      if (stats) {
        stats->numInvalid += rawErr.has_value() ? 0 : 1;
        stats->numPrevInvalid += prevInvalid ? 1 : 0;
      }
      if (makeComparableWithStored) {
        if (prevInvalid) {
          return {}; // force comparable value
        } else if (!rawErr.has_value()) {
          return prevValue;
        }
      }
    }
    return unpackingOptional(
        rawErr,
        [i, this](const auto& e) {
          return std::optional<double>(getLoss(i).val(squaredError(i, e)) * 0.5);
        },
        [] { return std::optional<double>{}; });
  }

  virtual std::optional<double> varGradHess(
      int64_t k,
      int v,
      Eigen::Ref<Eigen::VectorXd> grad,
      Eigen::Ref<Eigen::MatrixXd> hess) const override {
    auto& factor = std::get<0>(boundFactors[k]);
    auto& args = std::get<1>(boundFactors[k]);
    const auto& loss = getLoss(k);

    return withStaticRange<0, sizeof...(Variables)>(
        [&](auto... iWrapIndices) -> std::optional<double> {
          const int vTangentDim =
              (... +
               (decltype(iWrapIndices)::value != v
                    ? 0
                    : std::get<decltype(iWrapIndices)::value>(args)->getTangentDim()));
          Eigen::Map<Eigen::Matrix<double, ErrorSize, Eigen::Dynamic>> vJac(
              (double*)alloca(sizeof(double) * ErrorSize * vTangentDim), ErrorSize, vTangentDim);

          // Compute residual error and jacobians
          RawErrorType rawErr = factor(
              std::get<decltype(iWrapIndices)::value>(args)->value...,
              decltype(iWrapIndices)::value != v
                  ? NullRef(
                        ErrorSize, std::get<decltype(iWrapIndices)::value>(args)->getTangentDim())
                  : Eigen::Ref<Eigen::Matrix<double, ErrorSize, Variables::TangentDimSpec>>(
                        vJac)...);
          if constexpr (HasOptionalErrors) {
            if (!rawErr.has_value()) {
              return {};
            }
          }
          const ErrorType& err = unpackedOpt(rawErr);
          auto [softErr, dSoftErr] = loss.jet2(squaredError(k, err));

          Eigen::Map<Eigen::Matrix<double, ErrorSize, Eigen::Dynamic>> vAdjJac(
              (double*)alloca(sizeof(double) * ErrorSize * vTangentDim), ErrorSize, vTangentDim);
          if constexpr (HasPrecisionMatrix) {
            vAdjJac.noalias() = std::get<PrecisionMatrixIndex>(boundFactors[k]) * vJac;
          } else {
            vAdjJac.noalias() = vJac;
          }
          vAdjJac *= dSoftErr;

          grad.noalias() = err.transpose() * vAdjJac;
          hess.noalias() = vAdjJac.transpose() * vJac;

          return softErr * 0.5;
        });
  }

  std::tuple<double, double, double> singleExpectedDelta(
      int64_t k,
      const double* stepData,
      const BaSpaCho::PermutedCoalescedAccessor& acc) const override {
    auto& factor = std::get<0>(boundFactors[k]);
    auto& args = std::get<1>(boundFactors[k]);
    const auto& loss = getLoss(k);

    return withStaticRange<0, sizeof...(Variables)>(
        [&](auto... iWrapIndices) -> std::tuple<double, double, double> {
          std::tuple<Eigen::Map<Eigen::Matrix<double, ErrorSize, Variables::TangentDimSpec>>...>
          jacobians(Eigen::Map<Eigen::Matrix<double, ErrorSize, Variables::TangentDimSpec>>(
              std::get<decltype(iWrapIndices)::value>(args)->isSetToConstant()
                  ? nullptr
                  : (double*)alloca(
                        sizeof(double) * ErrorSize *
                        std::get<decltype(iWrapIndices)::value>(args)->getTangentDim()),
              ErrorSize,
              std::get<decltype(iWrapIndices)::value>(args)->getTangentDim())...);

          // Compute residual error and jacobians
          RawErrorType rawErr = factor(
              std::get<decltype(iWrapIndices)::value>(args)->value...,
              std::get<decltype(iWrapIndices)::value>(jacobians)...);
          if constexpr (HasOptionalErrors) {
            if (!rawErr.has_value()) {
              return {-1.0, 0.0, 0.0};
            }
          }
          ErrorType err = unpackedOpt(rawErr);

          // this is a workaround to a bug of clang "reference to local binding 'dSoftErr'
          // declared in enclosing function", triggered because of capture below.
          auto [softErr, dSoftErr_] = loss.jet2(squaredError(k, err));
          double dSoftErr = dSoftErr_;

          // adjust with soft loss, and possibly H
          err *= dSoftErr;
          if constexpr (HasPrecisionMatrix) {
            err = std::get<2>(boundFactors[k]) * err;
          }

          double delta = 0, gradSqNorm = 0;
          forEach<0, sizeof...(Variables)>([&](auto iWrap) {
            static constexpr int i = decltype(iWrap)::value;
            int64_t iIndex = std::get<i>(args)->index;
            if (iIndex == kConstantVar) {
              return;
            }

            static constexpr int iTangentDimSpec =
                std::remove_reference_t<decltype(*std::get<i>(args))>::TangentDimSpec;
            const int iTangentDim = std::get<i>(args)->getTangentDim();
            const auto& iJac = std::get<i>(jacobians);

            // gradient contribution
            Eigen::Map<Eigen::Vector<double, iTangentDimSpec>> gradContrib(
                (double*)alloca(sizeof(double) * iTangentDim), iTangentDim);
            gradContrib.noalias() = err.transpose() * iJac;

            // step segment
            int64_t paramStart = acc.paramStart(iIndex);
            Eigen::Map<const Eigen::Vector<double, iTangentDimSpec>> stepSeg(
                stepData + paramStart, iTangentDim);

            delta += stepSeg.dot(gradContrib);
            gradSqNorm += gradContrib.squaredNorm();
          });
          return {softErr * 0.5, delta, std::sqrt(gradSqNorm)};
        });
  }

  // (helper) compute single grad/hess, using atomically/simply according to `Ops`
  template <typename Ops>
  double computeSingleGradHess(
      int64_t k,
      double* gradData,
      const BaSpaCho::PermutedCoalescedAccessor& acc,
      double* hessData,
      bool updateCachedResults,
      bool dontRetryFailed) const {
    auto& factor = std::get<0>(boundFactors[k]);
    auto& args = std::get<1>(boundFactors[k]);
    const auto& loss = getLoss(k);

    if constexpr (HasOptionalErrors) {
      if (dontRetryFailed && std::get<OptionalErrorIndex>(boundFactors[k]).value < 0.0) {
        return 0.0;
      }
    }

    return withStaticRange<0, sizeof...(Variables)>([&](auto... iWrapIndices) {
      std::tuple<Eigen::Map<Eigen::Matrix<double, ErrorSize, Variables::TangentDimSpec>>...>
      jacobians(Eigen::Map<Eigen::Matrix<double, ErrorSize, Variables::TangentDimSpec>>(
          std::get<decltype(iWrapIndices)::value>(args)->isSetToConstant()
              ? nullptr
              : (double*)alloca(
                    sizeof(double) * ErrorSize *
                    std::get<decltype(iWrapIndices)::value>(args)->getTangentDim()),
          ErrorSize,
          std::get<decltype(iWrapIndices)::value>(args)->getTangentDim())...);

      // Compute residual error and jacobians
      const RawErrorType rawErr = factor(
          std::get<decltype(iWrapIndices)::value>(args)->value...,
          std::get<decltype(iWrapIndices)::value>(jacobians)...);
      if constexpr (HasOptionalErrors) {
        if (!rawErr.has_value()) {
          if (updateCachedResults || dontRetryFailed) {
            std::get<OptionalErrorIndex>(boundFactors[k]).value = -1.0; // force 0 when comparing
          }
          return 0.0;
        }
      }
      const ErrorType& err = unpackedOpt(rawErr);

      // this is a workaround to a bug of clang "reference to local binding 'dSoftErr'
      // declared in enclosing function", triggered because of capture below.
      auto [softErr, dSoftErr_] = loss.jet2(squaredError(k, err));
      auto& dSoftErr = dSoftErr_;

      forEach<0, sizeof...(Variables)>([&](auto iWrap) {
        static constexpr int i = decltype(iWrap)::value;
        int64_t iIndex = std::get<i>(args)->index;
        if (iIndex == kConstantVar) {
          return;
        }

        static constexpr int iTangentDimSpec =
            std::remove_reference_t<decltype(*std::get<i>(args))>::TangentDimSpec;
        const int iTangentDim = std::get<i>(args)->getTangentDim();
        const auto& iJac = std::get<i>(jacobians);
        Eigen::Map<Eigen::Matrix<double, ErrorSize, iTangentDimSpec>> iAdjJac(
            (double*)alloca(sizeof(double) * ErrorSize * iTangentDim), ErrorSize, iTangentDim);
        if constexpr (HasPrecisionMatrix) {
          iAdjJac.noalias() = std::get<PrecisionMatrixIndex>(boundFactors[k]) * iJac;
        } else {
          iAdjJac.noalias() = iJac;
        }
        iAdjJac *= dSoftErr;

        // gradient contribution
        int64_t paramStart = acc.paramStart(iIndex);
        Eigen::Map<Eigen::Vector<double, iTangentDimSpec>> gradSeg(
            gradData + paramStart, iTangentDim);
        Eigen::Map<Eigen::Vector<double, iTangentDimSpec>> gradSegAdd(
            (double*)alloca(sizeof(double) * iTangentDim), iTangentDim);
        gradSegAdd.noalias() = err.transpose() * iAdjJac;
        Ops::vectorAdd(gradSeg, gradSegAdd);

        if (hessData) {
          forEach<0, i>([&](auto jWrap) {
            static constexpr int j = decltype(jWrap)::value;
            int64_t jIndex = std::get<j>(args)->index;
            if (jIndex == kConstantVar) {
              return;
            }

            // Hessian off-diagonal contribution
            static constexpr int jTangentDimSpec =
                std::remove_reference_t<decltype(*std::get<j>(args))>::TangentDimSpec;
            const int jTangentDim = std::get<j>(args)->getTangentDim();
            const auto& jJac = std::get<j>(jacobians);
            auto odBlock = acc.block(hessData, iIndex, jIndex);
            Eigen::Map<Eigen::Matrix<double, iTangentDimSpec, jTangentDimSpec>> odBlockAdd(
                (double*)alloca(sizeof(double) * iTangentDim * jTangentDim),
                iTangentDim,
                jTangentDim);
            odBlockAdd.noalias() = iAdjJac.transpose() * jJac;
            Ops::matrixAdd(odBlock, odBlockAdd);
          });

          // Hessian diagonal contribution
          auto dBlock = acc.diagBlock(hessData, iIndex);
          Eigen::Map<Eigen::Matrix<double, iTangentDimSpec, iTangentDimSpec>> dBlockAdd(
              (double*)alloca(sizeof(double) * iTangentDim * iTangentDim),
              iTangentDim,
              iTangentDim);
          dBlockAdd.noalias() = iAdjJac.transpose() * iJac;
          Ops::matrixAdd(dBlock, dBlockAdd);
        }
      });

      double ret = softErr * 0.5;
      if constexpr (HasOptionalErrors) {
        if (updateCachedResults) {
          std::get<OptionalErrorIndex>(boundFactors[k]).value = ret;
        }
      }
      return ret;
    });
  }

  // [opaque] compute the sum of all costs
  virtual double computeCost(
      bool makeComparableWithStored,
      CostStats* stats = nullptr,
      dispenso::ThreadPool* threadPool = nullptr) const override {
    if (threadPool) { // multi-threaded
      std::vector<CostAndStats> perThreadCostAndStats;
      dispenso::TaskSet taskSet(*threadPool);
      dispenso::parallel_for(
          taskSet,
          perThreadCostAndStats,
          []() -> CostAndStats { return {0.0, CostStats{}}; },
          dispenso::makeChunkedRange(
              0L, boundFactors.size(), boundFactors.size() / threadPool->numThreads()),
          [this, makeComparableWithStored](
              CostAndStats& threadCostAndStats, int64_t iBegin, int64_t iEnd) {
            for (int64_t i = iBegin; i < iEnd; i++) {
              threadCostAndStats.first +=
                  computeSingleCost(i, makeComparableWithStored, &threadCostAndStats.second)
                      .value_or(0.0);
            }
          });
      const auto aggregatedCostAndStats = aggregate(perThreadCostAndStats);
      if (stats) {
        *stats = aggregatedCostAndStats.second;
      }
      return aggregatedCostAndStats.first;
    } else { // single-threaded
      double retv = 0;
      CostStats computedStats;
      for (size_t i = 0; i < boundFactors.size(); i++) {
        retv += computeSingleCost(i, makeComparableWithStored, &computedStats).value_or(0.0);
      }
      if (stats) {
        *stats = computedStats;
      }
      return retv;
    }
  }

  // [opaque] compute the sum of all costs, and add contribution to gradient/hessian
  virtual double computeGradHess(
      double* gradData,
      const BaSpaCho::PermutedCoalescedAccessor& acc,
      double* hessData,
      bool updateCachedResults,
      bool dontRetryFailed,
      dispenso::ThreadPool* threadPool = nullptr) const override {
    if (threadPool) { // multi-threaded
      std::vector<double> perThreadCost;
      dispenso::TaskSet taskSet(*threadPool);
      dispenso::parallel_for(
          taskSet,
          perThreadCost,
          []() -> double { return 0.0; },
          dispenso::makeChunkedRange(0L, boundFactors.size()),
          [=, this, &acc](double& threadCost, int64_t iBegin, int64_t iEnd) {
            for (int64_t i = iBegin; i < iEnd; i++) {
              threadCost += computeSingleGradHess<LockedSharedOps>(
                  i, gradData, acc, hessData, updateCachedResults, dontRetryFailed);
            }
          });
      return Eigen::Map<Eigen::VectorXd>(perThreadCost.data(), perThreadCost.size()).sum();
    } else { // single-threaded
      double retv = 0;
      for (size_t i = 0; i < boundFactors.size(); i++) {
        retv += computeSingleGradHess<PlainOps>(
            i, gradData, acc, hessData, updateCachedResults, dontRetryFailed);
      }
      return retv;
    }
  }

  // [opaque] ensure that all variables are registered
  virtual void registerVariables(
      std::vector<int64_t>& paramSizes,
      TypedStore<VariableStoreBase>& varStores) override {
    forEach<0, sizeof...(Variables)>([&, this](auto iWrap) {
      static constexpr int i = decltype(iWrap)::value;
      using Variable =
          std::remove_reference_t<decltype(*std::get<i>(std::get<1>(boundFactors[0])))>;
      auto& store = varStores.get<VariableStore<Variable>>();

      for (auto& tup : boundFactors) {
        auto& args = std::get<1>(tup);

        // register variable (if needed)
        auto& Vi = *std::get<i>(args);
        if (Vi.index == kUnsetIndex) {
          Vi.index = paramSizes.size();
          paramSizes.push_back(Vi.getTangentDim());
          store.variables.push_back(&Vi);
        }
      }
    });
  }

  // [opaque] collect blocks that we will have to write into
  virtual void registerBlocks(
      std::unordered_set<std::pair<int64_t, int64_t>, pair_hash>& blocks) override {
    forEach<0, sizeof...(Variables)>([&, this](auto iWrap) {
      static constexpr int i = decltype(iWrap)::value;

      for (auto& tup : boundFactors) {
        auto& args = std::get<1>(tup);

        // add off-diagonal block to Hessian structure (for registered ie non-const vars)
        auto& Vi = *std::get<i>(args);
        BASPACHO_CHECK(!Vi.isUnset());
        if (!Vi.isSetToConstant()) {
          forEach<0, i>([&](auto jWrap) {
            static constexpr int j = decltype(jWrap)::value;
            auto& Vj = *std::get<j>(args);
            if (!Vj.isSetToConstant()) {
              int64_t minIndex = std::min(Vi.index, Vj.index);
              int64_t maxIndex = std::max(Vi.index, Vj.index);
              blocks.insert(std::make_pair(maxIndex, minIndex));
            }
          });
        }
      }
    });
  }

  std::vector<TupleType> boundFactors;
};

} // namespace small_thing
