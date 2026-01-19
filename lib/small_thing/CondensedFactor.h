/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <small_thing/Factor.h>
#include <map>

namespace small_thing {

struct ProxyBase {};

struct DirectForwardBase : ProxyBase {};

struct LocalVar {
  int64_t globalIndex;
  int64_t tangentDim;
  int64_t localOffset;
};

inline int64_t findLocalIndex(const std::vector<LocalVar>& localVars, int64_t globalIndex) {
  auto it = std::lower_bound(
      localVars.begin(), localVars.end(), globalIndex, [](const LocalVar& v, int64_t g) -> bool {
        return v.globalIndex < g;
      });
  BASPACHO_CHECK(it != localVars.end());
  BASPACHO_CHECK_EQ(it->globalIndex, globalIndex);
  return std::distance(localVars.begin(), it);
}

// Implementation of factor store for "condensed" factors
template <typename... Proxies>
class CondensedFactorStore : public FactorStoreBase {
 public:
  using ProxyTuple = std::tuple<Proxies...>;
  struct Entry {
    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    double c;
    ProxyTuple proxies;

    std::vector<LocalVar> localVars;
    int64_t totalVarTgDim;

    // constructor
    template <typename DerivedH, typename DerivedB>
    Entry(
        const Eigen::MatrixBase<DerivedH>& H,
        const Eigen::MatrixBase<DerivedB>& b,
        double c,
        Proxies&&... ps)
        : H(H), b(b), c(c), proxies(std::forward<Proxies>(ps)...) {
      BASPACHO_CHECK_EQ(H.rows(), H.cols());
      BASPACHO_CHECK_EQ(H.rows(), b.size());

      int64_t totalProxyResultDim = 0;
      forEach<0, sizeof...(Proxies)>([&](auto iWrap) {
        static constexpr int i = decltype(iWrap)::value;
        const auto& p = std::get<i>(proxies);
        using Proxy = std::remove_cv_t<std::remove_reference_t<decltype(p)>>;
        if constexpr (std::is_base_of_v<DirectForwardBase, Proxy>) {
          const int64_t numVars = p.numVariables();
          for (int64_t v = 0; v < numVars; v++) {
            auto& Vj = p.getVar(v);
            // considering outputs - therefore add even if it's constant / unset
            totalProxyResultDim += Vj.getTangentDim();
          }
        } else {
          const int64_t numEntries = p.numEntries();
          totalProxyResultDim += numEntries * Proxy::ResultDim;
        }
      });
      BASPACHO_CHECK_EQ(H.rows(), totalProxyResultDim);
    }

    static bool verifyAllJacobians(
        const std::vector<Entry>& entries,
        double epsilon,
        double relativeTolerance,
        double absoluteTolerance,
        bool stopAtFirstError,
        const LogFunc& log) {
      std::vector<std::vector<Eigen::VectorXd>> maxRelErr(sizeof...(Proxies));
      double paramMaxRelErrs = 0.0;
      int maxRelErrCol = -1;
      bool foundError = false;
      static constexpr int64_t kMaxCheck = 100;

      // iterate on proxy types
      forEach<0, sizeof...(Proxies)>([&](auto iWrap) {
        static constexpr int i = decltype(iWrap)::value;
        using Proxy = std::remove_cv_t<
            std::remove_reference_t<decltype(std::get<i>(std::declval<std::tuple<Proxies...>>()))>>;

        // nothing to verify in case of direct forwarding, so just skip it
        if constexpr (!std::is_base_of_v<DirectForwardBase, Proxy>) {
          using InVarsTypes = typename Proxy::InputVariablesType;
          static constexpr int numIndices = std::tuple_size_v<InVarsTypes>;

          maxRelErr[i].resize(numIndices);

          // iterate on proxy's input var types
          int64_t nCheck = 0, total = 0;
          forEach<0, numIndices>([&](auto jWrap) {
            static constexpr int j = decltype(jWrap)::value;
            using Variable =
                std::remove_reference_t<decltype(*std::get<j>(std::declval<InVarsTypes>()))>;
            using jVarSpec = VarSpec<typename Variable::DataType>;
            static constexpr int jTangentDim = jVarSpec::TangentDimSpec;
            static constexpr int jDataDim = jVarSpec::DataDimSpec;
            static_assert(jTangentDim != Eigen::Dynamic, "Not supported!");
            static_assert(jDataDim != Eigen::Dynamic, "Not supported!");

            maxRelErr[i][j].setZero(jTangentDim);

            nCheck = 0;
            total = 0;
            for (const Entry& entry : entries) {
              const auto& p = std::get<i>(entry.proxies);
              const int64_t numEntries = p.numEntries();
              total += numEntries;
              if (nCheck >= kMaxCheck) {
                continue;
              }

              for (int64_t e = 0; e < numEntries; e++) {
                if (nCheck >= kMaxCheck) {
                  break;
                }
                auto& jVar = *std::get<j>(p.inputVariables(e));
                typename Proxy::JacobianType J;
                typename Proxy::ResultType val = p.eval(e, &J);
                Eigen::Matrix<double, Proxy::ResultDim, jTangentDim> jJac = std::get<j>(J);
                Eigen::Matrix<double, Proxy::ResultDim, jTangentDim> jNumJac;

                for (int d = 0; d < jTangentDim; d++) {
                  Eigen::Vector<double, jDataDim> jVarBackup;
                  jVarSpec::getData(jVar.value, jVarBackup);

                  Eigen::Vector<double, jTangentDim> tgStep =
                      Eigen::Vector<double, jTangentDim>::Zero();
                  tgStep[d] = epsilon;
                  jVarSpec::applyBoxPlus(jVar.value, tgStep);
                  typename Proxy::ResultType pVal = p.eval(e);
                  jVarSpec::setData(jVar.value, jVarBackup); // restore

                  jNumJac.col(d) = (pVal - val) / epsilon;
                  double relErr =
                      std::max((jNumJac.col(d) - jJac.col(d)).norm() - absoluteTolerance, 0.0) /
                      (jNumJac.col(d).norm() + epsilon);
                  maxRelErr[i][j][d] = std::max(relErr, maxRelErr[i][j][d]);
                  if (relErr > paramMaxRelErrs) {
                    paramMaxRelErrs = relErr;
                    maxRelErrCol = d;
                  }
                }

                if (paramMaxRelErrs > relativeTolerance) {
                  if (log && stopAtFirstError) {
                    std::stringstream ss;
                    ss << prettyTypeName<Proxy>() << ".Jac" << j << ":\n"
                       << jJac << "\nwhile numeric Jacobian is\n"
                       << jNumJac << "\n and has relative error " << paramMaxRelErrs << " > "
                       << relativeTolerance << " in column " << maxRelErrCol;
                    log(ss.str());
                  }
                  foundError = true;
                }

                nCheck++;
                if (foundError && stopAtFirstError) {
                  break;
                }
              }
              if (foundError && stopAtFirstError) {
                break;
              }
            }
            if (foundError && stopAtFirstError) {
              return;
            }
          });

          if (log) {
            std::stringstream ss;
            ss << "Verified proxy class:\n  " << prettyTypeName<Proxy>() << "\n"
               << "Factors checked: " << nCheck << "/" << total << ", Jacobian check "
               << (foundError ? "FAILED" : "OK!") << " (relative tolerance: " << relativeTolerance
               << ", absolute tolerance: " << absoluteTolerance << ")\n";
            for (size_t j = 0; j < maxRelErr[i].size(); j++) {
              ss << "Relative errors in cols of " << j << "-th Jacobian:\n  "
                 << maxRelErr[i][j].transpose() << std::endl;
            };
            log(ss.str());
          }
        }
      });

      return !foundError;
    }

    static void registerAllVariables(
        const std::vector<Entry>& entries,
        std::vector<int64_t>& paramSizes,
        TypedStore<VariableStoreBase>& varStores) {
      // iterate on proxy types
      forEach<0, sizeof...(Proxies)>([&](auto iWrap) {
        static constexpr int i = decltype(iWrap)::value;
        using Proxy = std::remove_cv_t<
            std::remove_reference_t<decltype(std::get<i>(std::declval<std::tuple<Proxies...>>()))>>;
        if constexpr (std::is_base_of_v<DirectForwardBase, Proxy>) {
          using Variable = typename Proxy::Var;
          auto& store = varStores.get<VariableStore<Variable>>();

          for (const Entry& entry : entries) {
            const auto& p = std::get<i>(entry.proxies);
            const int64_t numVars = p.numVariables();

            for (int64_t v = 0; v < numVars; v++) {
              // register variable (if needed)
              auto& Vj = p.getVar(v);
              if (Vj.index == kUnsetIndex) {
                Vj.index = paramSizes.size();
                paramSizes.push_back(Vj.getTangentDim());
                store.variables.push_back(&Vj);
              }
            }
          }
        } else {
          using InVarsTypes = typename Proxy::InputVariablesType;
          static constexpr int numIndices = std::tuple_size_v<InVarsTypes>;

          // iterate on proxy's input var types
          forEach<0, numIndices>([&](auto jWrap) {
            static constexpr int j = decltype(jWrap)::value;
            using Variable =
                std::remove_reference_t<decltype(*std::get<j>(std::declval<InVarsTypes>()))>;
            static constexpr int tangentDim = VarSpec<typename Variable::DataType>::TangentDimSpec;
            auto& store = varStores.get<VariableStore<Variable>>();

            for (const Entry& entry : entries) {
              const auto& p = std::get<i>(entry.proxies);
              const int64_t numEntries = p.numEntries();
              for (int64_t e = 0; e < numEntries; e++) {
                // register variable (if needed)
                auto& Vj = *std::get<j>(p.inputVariables(e));
                if (Vj.index == kUnsetIndex) {
                  Vj.index = paramSizes.size();
                  paramSizes.push_back(tangentDim);
                  store.variables.push_back(&Vj);
                }
              }
            }
          });
        }
      });
    }

    void init() {
      // re-compute if needed, must update if indices changed, or some var was set const/non const
      localVars.clear();

      // not unordered, so we keep sorted by variable index
      std::map<int64_t, int64_t> varIndexToTgDim;

      // iterate on proxy types
      forEach<0, sizeof...(Proxies)>([&](auto iWrap) {
        static constexpr int i = decltype(iWrap)::value;
        const auto& p = std::get<i>(proxies);
        using Proxy = std::remove_cv_t<std::remove_reference_t<decltype(p)>>;
        if constexpr (std::is_base_of_v<DirectForwardBase, Proxy>) {
          const int64_t numVars = p.numVariables();

          for (int64_t v = 0; v < numVars; v++) {
            auto& Vj = p.getVar(v);
            BASPACHO_CHECK(Vj.index != kUnsetIndex);
            if (!Vj.isSetToConstant()) {
              varIndexToTgDim[Vj.index] = Vj.getTangentDim();
            }
          }
        } else {
          using InVarsTypes = typename Proxy::InputVariablesType;
          static constexpr int numIndices = std::tuple_size_v<InVarsTypes>;

          const int64_t numEntries = p.numEntries();

          // iterate on proxy's input var types
          forEach<0, numIndices>([&](auto jWrap) {
            static constexpr int j = decltype(jWrap)::value;
            using Variable =
                std::remove_reference_t<decltype(*std::get<j>(std::declval<InVarsTypes>()))>;
            static constexpr int tangentDim = VarSpec<typename Variable::DataType>::TangentDimSpec;

            for (int64_t e = 0; e < numEntries; e++) {
              // register variable (if needed)
              auto& Vj = *std::get<j>(p.inputVariables(e));
              BASPACHO_CHECK(Vj.index != kUnsetIndex);
              if (!Vj.isSetToConstant()) {
                varIndexToTgDim[Vj.index] = tangentDim;
              }
            }
          });
        }
      });

      localVars.reserve(varIndexToTgDim.size());

      // collect block info to build J (to do H -> J.T * H * J at runtime)
      int64_t localOffset = 0;
      for (auto [varIndex, tgDim] : varIndexToTgDim) {
        localVars.push_back(LocalVar{
            .globalIndex = varIndex,
            .tangentDim = tgDim,
            .localOffset = localOffset,
        });
        localOffset += tgDim;
      }
      totalVarTgDim = localOffset;
    }

    // compute the concatenation of all proxy values
    void computeX(Eigen::VectorXd& x) const {
      x.resize(H.rows());

      int64_t proxyResultOffset = 0;
      forEach<0, sizeof...(Proxies)>([&](auto iWrap) {
        static constexpr int i = decltype(iWrap)::value;
        const auto& p = std::get<i>(proxies);
        using Proxy = std::remove_cv_t<std::remove_reference_t<decltype(p)>>;
        if constexpr (std::is_base_of_v<DirectForwardBase, Proxy>) {
          const int64_t numVars = p.numVariables();
          for (int64_t v = 0; v < numVars; v++) {
            auto& jVar = p.getVar(v);
            p.eval(v, x.segment(proxyResultOffset, jVar.getTangentDim()));
            proxyResultOffset += jVar.getTangentDim();
          }
        } else {
          const int64_t numEntries = p.numEntries();
          for (int64_t e = 0; e < numEntries; e++) {
            x.segment<Proxy::ResultDim>(proxyResultOffset) = p.eval(e);
            proxyResultOffset += Proxy::ResultDim;
          }
        }
      });
    }

    // compute the concatenation of all proxy values
    void computeXandJ(Eigen::VectorXd& x, Eigen::MatrixXd& J) const {
      x.resize(H.rows());
      J.setZero(H.rows(), totalVarTgDim);

      int64_t proxyResultOffset = 0;
      forEach<0, sizeof...(Proxies)>([&](auto iWrap) {
        static constexpr int i = decltype(iWrap)::value;
        const auto& p = std::get<i>(proxies);
        using Proxy = std::remove_cv_t<std::remove_reference_t<decltype(p)>>;
        if constexpr (std::is_base_of_v<DirectForwardBase, Proxy>) {
          const int64_t numVars = p.numVariables();
          for (int64_t v = 0; v < numVars; v++) {
            auto& jVar = p.getVar(v);
            const int jTgDim = jVar.getTangentDim();
            p.eval(v, x.segment(proxyResultOffset, jTgDim));
            if (!jVar.isSetToConstant()) {
              const int64_t localIndex = findLocalIndex(localVars, jVar.index);
              J.block(proxyResultOffset, localVars[localIndex].localOffset, jTgDim, jTgDim)
                  .setIdentity();
            }
            proxyResultOffset += jTgDim;
          }
        } else {
          using InVarsTypes = typename Proxy::InputVariablesType;
          static constexpr int numIndices = std::tuple_size_v<InVarsTypes>;
          const int64_t numEntries = p.numEntries();
          for (int64_t e = 0; e < numEntries; e++) {
            typename Proxy::JacobianType pJ;
            x.segment<Proxy::ResultDim>(proxyResultOffset) = p.eval(e, &pJ);

            // add blocks into J (TODO: J is sparse, optimize one day)
            const auto inputVars = p.inputVariables(e);
            forEach<0, numIndices>([&](auto jWrap) {
              static constexpr int j = decltype(jWrap)::value;
              const auto& jVar = *std::get<j>(inputVars);
              if (jVar.isSetToConstant()) {
                return;
              }
              const int64_t localIndex = findLocalIndex(localVars, jVar.index);
              J.block(
                  proxyResultOffset,
                  localVars[localIndex].localOffset,
                  Proxy::ResultDim,
                  localVars[localIndex].tangentDim) = std::get<j>(pJ);
            });

            proxyResultOffset += Proxy::ResultDim;
          }
        }
      });
    }
  };

  virtual ~CondensedFactorStore() override {}

  virtual int64_t totalErrorDimensionality() const override {
    throw std::runtime_error("CondensedFactor: totalErrorDimensionality is not supported");
  }

  virtual int64_t numCosts() const override {
    return entries.size();
  }

  std::optional<double> computeSingleCost(int64_t i) const {
    Eigen::VectorXd x; // TODO: avoid continuous alloc/free
    const Entry& entry = entries[i];
    entry.computeX(x);
    return 0.5 * x.dot(entry.H * x) + entry.b.dot(x) + entry.c;
  }

  virtual std::optional<double> varGradHess(
      int64_t /* k */,
      int /* v */,
      Eigen::Ref<Eigen::VectorXd> /* grad */,
      Eigen::Ref<Eigen::MatrixXd> /* hess */) const override {
    throw std::runtime_error("CondensedFactor: varGradHess is not supported");
  }

  std::tuple<double, double, double> singleExpectedDelta(
      int64_t /* k */,
      const double* /* stepData */,
      const BaSpaCho::PermutedCoalescedAccessor& /* acc */) const override {
    throw std::runtime_error("CondensedFactor: singleExpectedDelta is not supported");
  }

  double computeSingleGradHess(
      int64_t k,
      double* gradData,
      const BaSpaCho::PermutedCoalescedAccessor& acc,
      double* hessData) const {
    Eigen::VectorXd x;
    Eigen::MatrixXd J; // TODO: avoid continuous alloc/free
    const Entry& entry = entries[k];
    entry.computeXandJ(x, J);

    // in variable's tangent space (TODO: avoid allocations)
    Eigen::VectorXd Jtb = J.transpose() * entry.b;
    Eigen::MatrixXd JtH = J.transpose() * entry.H;
    Eigen::MatrixXd JtHJ = JtH * J;
    Eigen::VectorXd grad = Jtb + JtH * x;

    // write blocks to hessian and gradient (TODO: lock just here)
    for (int64_t i = 0; i < (int64_t)entry.localVars.size(); i++) {
      const int64_t iIndex = entry.localVars[i].globalIndex;
      if (iIndex == kConstantVar) {
        continue;
      }
      const int64_t iLocalOffset = entry.localVars[i].localOffset;
      const int64_t iTangentDim = entry.localVars[i].tangentDim;
      const int64_t paramStart = acc.paramStart(iIndex);

      // gradient contribution
      Eigen::Map<Eigen::VectorXd> gradSeg(gradData + paramStart, iTangentDim);
      gradSeg += grad.segment(iLocalOffset, iTangentDim);

      if (hessData) {
        for (int64_t j = 0; j <= i; j++) {
          const int64_t jIndex = entry.localVars[j].globalIndex;
          if (jIndex == kConstantVar) {
            continue;
          }
          const int64_t jLocalOffset = entry.localVars[j].localOffset;
          const int64_t jTangentDim = entry.localVars[j].tangentDim;

          // hessian contribution (TODO: diagonal blocks have a faster accessor)
          auto block = acc.block(hessData, iIndex, jIndex);
          block += JtHJ.block(iLocalOffset, jLocalOffset, iTangentDim, jTangentDim);
        }
      }
    }
    // return cost value
    return 0.5 * x.dot(entry.H * x) + entry.b.dot(x) + entry.c;
  }

  virtual bool unweightedError(int64_t, Eigen::Ref<Eigen::VectorXd>) const override {
    throw std::runtime_error("CondensedFactor: unweightedError is not supported");
  }

  virtual double unweightedSquaredError(int64_t) const override {
    throw std::runtime_error("CondensedFactor: unweightedSquaredError is not supported");
  }

  virtual double covWeightedSquaredError(int64_t) const override {
    throw std::runtime_error("CondensedFactor: covWeightedSquaredError is not supported");
  }

  virtual std::optional<double> singleCost(int64_t i) const override {
    return computeSingleCost(i);
  }

  virtual VarBase* costVar(int64_t, int) const override {
    throw std::runtime_error("CondensedFactor: costVar is not supported");
  }

  virtual std::string name() const override {
    return prettyTypeName<CondensedFactorStore>();
  }

  virtual bool verifyJacobians(
      double epsilon,
      double relativeTolerance,
      double absoluteTolerance,
      bool stopAtFirstError,
      const LogFunc& log) const override {
    return Entry::verifyAllJacobians(
        entries, epsilon, relativeTolerance, absoluteTolerance, stopAtFirstError, log);
  }

  virtual double computeCost(
      bool /* makeComparableWithStored */,
      CostStats* stats,
      dispenso::ThreadPool* /* threadPool */) const override {
    double totalCost = 0.0;
    for (int64_t i = 0; i < (int64_t)entries.size(); i++) {
      totalCost += computeSingleCost(i).value_or(0.0); // TODO: threads
    }
    if (stats) {
      stats->numTotal += entries.size();
    }
    return totalCost;
  }

  virtual double computeGradHess(
      double* gradData,
      const BaSpaCho::PermutedCoalescedAccessor& acc,
      double* hessData,
      bool /* updateCachedResults */,
      bool /* dontRetryFailed */,
      dispenso::ThreadPool* /* threadPool */) const override {
    double totalCost = 0.0;
    for (int64_t i = 0; i < (int64_t)entries.size(); i++) {
      totalCost += computeSingleGradHess(i, gradData, acc, hessData); // TODO: threads
    }
    return totalCost;
  }

  virtual void registerVariables(
      std::vector<int64_t>& paramSizes,
      TypedStore<VariableStoreBase>& varStores) override {
    Entry::registerAllVariables(entries, paramSizes, varStores);
  }

  virtual void registerBlocks(
      std::unordered_set<std::pair<int64_t, int64_t>, pair_hash>& blocks) override {
    for (auto& entry : entries) {
      entry.init();

      for (int64_t i = 0; i < (int64_t)entry.localVars.size(); i++) {
        const int64_t iIndex = entry.localVars[i].globalIndex;
        for (int64_t j = 0; j < i; j++) {
          const int64_t jIndex = entry.localVars[j].globalIndex;
          const int64_t minIndex = std::min(iIndex, jIndex);
          const int64_t maxIndex = std::max(iIndex, jIndex);
          blocks.insert(std::make_pair(maxIndex, minIndex));
        }
      }
    }
  }

  std::vector<Entry> entries;
};

} // namespace small_thing
