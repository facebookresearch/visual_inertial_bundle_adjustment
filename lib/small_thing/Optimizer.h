/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <small_thing/CondensedFactor.h>
#include <small_thing/Factor.h>
#include <small_thing/PCG.h>
#include <small_thing/Preconditioner.h>
#include <iostream>

namespace BaSpaCho {
class Solver;
using SolverPtr = std::unique_ptr<Solver>;
} // namespace BaSpaCho

namespace small_thing {

using FactorDebugFunc = std::function<void(bool stepApplied, FactorStoreBase*, int)>;
using PreStepCallback = std::function<void(int)>;

class Optimizer {
 public:
  using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
  using Clock = std::chrono::high_resolution_clock;

  enum SolverType {
    Solver_Direct,
    Solver_PCG_Trivial,
    Solver_PCG_Jacobi,
    Solver_PCG_GaussSeidel,
    Solver_PCG_LowerPrecSolve,
  };

  // settings for optimize function
  struct Settings {
    int maxNumIterations = 50;
    unsigned int numThreads = 8;
    SolverType solverType = Solver_Direct;
    int pcgMaxIterations = 40;
    double pcgDesiredResidual = 1e-10;

    // tolerances used for stopping conditions
    double absoluteCostTolerance = 1e-8;
    double relativeCostTolerance = 1e-10;
    double variablesTolerance = 1e-5;

    // iteration settings
    int stopIfNoImprovementFor = 3;
    int distanceFromTroubledIteration = 3;
    double damping = 1e-5;
    double dampingAdjustOnFail = 2.5;
    double dampingAdjustOnGoodStep = 0.7;
    double dampingAdjustOnAverageStep = 1.5;
    double dampingMax = 1e8;
    double dampingMin = 1e-9;

    // desired cost reduction, relative to model cost decrease
    // if this is not obtained, smaller step multiples will be attempted
    double minRelativeCostReduction = 0.3;

    // when retrying smaller multiples, this factor will be applied
    double stepFactorDecrease = 0.3;

    // max number of attempts we will do
    int maxStepFactorAttempts = 2;

    // if enabled, after scaling by a factor, a "sub-step" is attempted.
    // this is useful in presence of non-linearities to move out of the
    // walls of a narrow "canyon" in the cost function
    bool trySubStep = true;

    // threshold to consider application of a factor of step a "good" step
    double minStepFactorForGood = 0.7;

    // logger function
    LogFunc log = [](const std::string& str) { std::cout << str << std::endl; };

    // trigger debugging of non-linearities at iteration = n
    int triggerDebuggingOfNonlinearities = -1;

    // for debugging of non-linearities (can eg set a global flag and debug print)
    FactorDebugFunc problematicFactorDebugFunc;

    // just a callback invoked before every step
    PreStepCallback preStepCallback;
  };

  struct Summary {
    double initialCost;
    double finalCost;
    int numTroubledSeqs;
    int largestTroubledSeq;
    int numIterations;
  };

  struct MarginalProblem {
    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    double cost;
    std::vector<int64_t> offsets;
  };

  struct DirectSolverData {
    BaSpaCho::SolverPtr solver;
    Eigen::VectorXd grad;
    Eigen::VectorXd hess;
    double cost;
  };

  template <
      typename Factor,
      typename Derived,
      typename SoftLoss,
      typename... Variables,
      std::enable_if_t<std::is_base_of_v<Loss, SoftLoss>, int> q = 0>
  auto addFactor(
      Factor&& f,
      const Eigen::MatrixBase<Derived>& precisionMatrix,
      const SoftLoss& l,
      Variables&... v) {
    using FactorStoreType = FactorStore<Factor, true, SoftLoss, Variables...>;
    static_assert((std::is_base_of_v<VarBase, Variables> && ...));
    auto& store = factorStores.get<FactorStoreType>();
    store.addFactor(std::forward<Factor>(f), std::make_tuple(&v...), precisionMatrix, &l);
    return std::pair<FactorStoreType*, int64_t>(&store, store.boundFactors.size() - 1);
  }

  template <
      typename Factor,
      typename Derived,
      typename Variable0,
      typename... Variables,
      std::enable_if_t<std::is_base_of_v<VarBase, Variable0>, int> q = 0>
  auto addFactor(
      Factor&& f,
      const Eigen::MatrixBase<Derived>& precisionMatrix,
      Variable0& v0,
      Variables&... v) {
    using FactorStoreType = FactorStore<Factor, true, TrivialLoss, Variable0, Variables...>;
    static_assert((std::is_base_of_v<VarBase, Variables> && ...));
    auto& store = factorStores.get<FactorStoreType>();
    store.addFactor(std::forward<Factor>(f), std::make_tuple(&v0, &v...), precisionMatrix);
    return std::pair<FactorStoreType*, int64_t>(&store, store.boundFactors.size() - 1);
  }

  template <
      typename Factor,
      typename SoftLoss,
      typename... Variables,
      std::enable_if_t<std::is_base_of_v<Loss, SoftLoss>, int> q = 0>
  auto addFactor(Factor&& f, const SoftLoss& l, Variables&... v) {
    using FactorStoreType = FactorStore<Factor, false, SoftLoss, Variables...>;
    static_assert((std::is_base_of_v<VarBase, Variables> && ...));
    auto& store = factorStores.get<FactorStoreType>();
    store.addFactor(std::forward<Factor>(f), std::make_tuple(&v...), &l);
    return std::pair<FactorStoreType*, int64_t>(&store, store.boundFactors.size() - 1);
  }

  template <
      typename Factor,
      typename Variable0,
      typename... Variables,
      std::enable_if_t<std::is_base_of_v<VarBase, Variable0>, int> q = 0>
  auto addFactor(Factor&& f, Variable0& v0, Variables&... v) {
    using FactorStoreType = FactorStore<Factor, false, TrivialLoss, Variable0, Variables...>;
    static_assert((std::is_base_of_v<VarBase, Variables> && ...));
    auto& store = factorStores.get<FactorStoreType>();
    store.addFactor(std::forward<Factor>(f), std::make_tuple(&v0, &v...));
    return std::pair<FactorStoreType*, int64_t>(&store, store.boundFactors.size() - 1);
  }

  template <
      typename DerivedH,
      typename DerivedB,
      typename Proxy0,
      typename... Proxies,
      std::enable_if_t<std::is_base_of_v<ProxyBase, Proxy0>, int> q = 0>
  auto addCondensedFactor(
      const Eigen::MatrixBase<DerivedH>& H,
      const Eigen::MatrixBase<DerivedB>& b,
      double c,
      Proxy0&& p0,
      Proxies&&... ps) {
    using FactorStoreType = CondensedFactorStore<Proxy0, Proxies...>;
    static_assert((std::is_base_of_v<ProxyBase, Proxies> && ...));
    auto& store = factorStores.get<FactorStoreType>();
    store.entries.emplace_back(H, b, c, std::forward<Proxy0>(p0), std::forward<Proxies>(ps)...);
    return std::pair<FactorStoreType*, int64_t>(&store, store.entries.size() - 1);
  }

  // register one variable, making sure it can be referenced by its `index`
  template <typename Variable>
  void registerVariable(Variable& v) {
    if (v.index == kUnsetIndex) {
      v.index = paramSizes.size();
      paramSizes.push_back(v.getTangentDim());
      variableStores.get<VariableStore<Variable>>().variables.push_back(&v);
    }
  }

  // clear all registered variables - allows to change variable const/non-const, and reuse factors
  void unregisterAllVariables();

  // ensure all variables referenced by factors are registered (and have an `index`, unless const)
  void registerAllVariables();

  // add currently registered variables to "elimination range" (ie make it a Schur block)
  void registeredVariablesToEliminationRange();

  int64_t totalVarDimensionality() const;

  int64_t totalErrorDimensionality() const;

  double computeGradHess(
      double* gradData,
      const BaSpaCho::PermutedCoalescedAccessor& acc,
      double* hessData,
      bool updateCachedResults,
      bool dontRetryFailed,
      dispenso::ThreadPool* threadPool = nullptr) const;

  bool verifyJacobians(
      double epsilon = 1e-7,
      double relativeTolerance = 1e-3,
      double absoluteTolerance = 1e-3,
      bool stopAtFirstError = false,
      const LogFunc& log = [](const std::string& str) { std::cout << str << std::endl; }) const;

  double computeCost(
      bool makeComparableWithStored = false,
      CostStats* stats = nullptr,
      dispenso::ThreadPool* threadPool = nullptr) const;

  int64_t variableBackupSize() const;

  void backupVariables(std::vector<double>& data) const;

  void restoreVariables(const std::vector<double>& data);

  // returns Linf, L2, L1 of vector with step-to-variable ratiost
  std::tuple<double, double, double> applyStep(
      const Eigen::VectorXd& step,
      const BaSpaCho::PermutedCoalescedAccessor& acc);

  struct ExpectedValues;

  void prepareExpectedValues(
      ExpectedValues& ev,
      const Eigen::VectorXd& step,
      const BaSpaCho::PermutedCoalescedAccessor& acc) const;

  void compareExpectedValues(ExpectedValues& ev) const;

  void addDamping(
      Eigen::VectorXd& hess,
      const BaSpaCho::PermutedCoalescedAccessor& acc,
      double lambda) const;

  static std::string solverToString(SolverType solverType);

  // creates a solver
  BaSpaCho::SolverPtr initSolver(
      int numThreads,
      bool fullElim = true,
      const std::unordered_set<int64_t>& paramsToReorderLast = {});

  // creates a "solve" function that will either
  // 1. invoke the direct solver
  // 2. apply partial elimination, run PCG, backtrack to a full solution
  std::pair<std::function<std::string()>, std::function<std::string(Eigen::VectorXd&)>>
  factorSolveFunctions(
      BaSpaCho::Solver& solver,
      Eigen::VectorXd& hess,
      const Settings& settings,
      int iterativeStart);

  void initDirectSolverData(
      DirectSolverData& ds,
      const Settings& settings,
      const std::unordered_set<int64_t>& paramsToReorderLast = {});

  // compute marginal on all elimination variables (AKA Schur complement)
  std::pair<Eigen::MatrixXd, std::vector<int64_t>> sparseElimMarginalInformation(
      const Settings& settings);

  // overload, default settings
  std::pair<Eigen::MatrixXd, std::vector<int64_t>> sparseElimMarginalInformation();

  // compute big marginal block H and error b. returns {H, b, offsets}
  MarginalProblem computeMarginalProblem(
      const Settings& settings,
      const std::vector<int64_t>& indices);

  // overload, default settings
  MarginalProblem computeMarginalProblem(const std::vector<int64_t>& indices);

  // compute update conditioning on a prescribed update on some variables
  void updateUnderConditioning(
      const Settings& settings,
      const std::unordered_map<int64_t, Eigen::VectorXd>& update);

  // overload, default settings
  void updateUnderConditioning(const std::unordered_map<int64_t, Eigen::VectorXd>& update);

  // returns a list of pairs {matrix, offsets}, which joint covariance and offsets for each
  // list or parameters `indices` in the list `indicesList`
  std::vector<std::pair<Eigen::MatrixXd, std::vector<int64_t>>> computeJointCovariances(
      const Settings& settings,
      const std::vector<std::vector<int64_t>>& indicesLists);

  // overload, default settings
  std::vector<std::pair<Eigen::MatrixXd, std::vector<int64_t>>> computeJointCovariances(
      const std::vector<std::vector<int64_t>>& indicesLists);

  // compute covariance block for variables in list `indices`, one per variable
  std::vector<Eigen::MatrixXd> computeCovariances(
      const Settings& settings,
      const std::vector<int64_t>& indices);

  // overload, default settings
  std::vector<Eigen::MatrixXd> computeCovariances(const std::vector<int64_t>& indices);

  Summary optimize();

  Summary optimize(const Settings& settings);

  TypedStore<FactorStoreBase> factorStores;
  TypedStore<VariableStoreBase> variableStores;
  std::vector<int64_t> paramSizes;
  std::vector<int64_t> elimRanges;
};

} // namespace small_thing
