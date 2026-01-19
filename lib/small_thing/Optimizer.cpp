/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
#include <baspacho/baspacho/Solver.h>
#include <small_thing/Optimizer.h>
#include <iomanip>

namespace small_thing {

void Optimizer::unregisterAllVariables() {
  for (const auto& [_, vStore] : variableStores.stores) {
    vStore->unregisterAll();
  }
  variableStores.stores.clear();

  elimRanges.clear();
  paramSizes.clear();
}

void Optimizer::registerAllVariables() {
  for (const auto& [_, fStore] : factorStores.stores) {
    fStore->registerVariables(paramSizes, variableStores);
  }
}

void Optimizer::registeredVariablesToEliminationRange() {
  if (paramSizes.size() == (elimRanges.empty() ? 0 : elimRanges.back())) {
    return; // don't add empty ranges
  }
  if (elimRanges.empty()) {
    elimRanges.push_back(0);
  }
  elimRanges.push_back(paramSizes.size());
}

int64_t Optimizer::totalVarDimensionality() const {
  int64_t sum = 0;
  for (const auto& [_, vStore] : variableStores.stores) {
    sum += vStore->totalVarDimensionality();
  }
  return sum;
}

int64_t Optimizer::totalErrorDimensionality() const {
  int64_t sum = 0;
  for (const auto& [_, fStore] : factorStores.stores) {
    sum += fStore->totalErrorDimensionality();
  }
  return sum;
}

double Optimizer::computeGradHess(
    double* gradData,
    const BaSpaCho::PermutedCoalescedAccessor& acc,
    double* hessData,
    bool updateCachedResults,
    bool dontRetryFailed,
    dispenso::ThreadPool* threadPool) const {
  double retv = 0.0;
  for (auto& [_, fStore] : factorStores.stores) {
    retv += fStore->computeGradHess(
        gradData, acc, hessData, updateCachedResults, dontRetryFailed, threadPool);
  }

  return retv;
}

bool Optimizer::verifyJacobians(
    double epsilon,
    double relativeTolerance,
    double absoluteTolerance,
    bool stopAtFirstError,
    const LogFunc& log) const {
  for (const auto& [_, fStore] : factorStores.stores) {
    if (!fStore->verifyJacobians(
            epsilon, relativeTolerance, absoluteTolerance, stopAtFirstError, log)) {
      return false;
    }
  }
  return true;
}

double Optimizer::computeCost(
    bool makeComparableWithStored,
    CostStats* stats,
    dispenso::ThreadPool* threadPool) const {
  double retv = 0.0;
  for (const auto& [_, fStore] : factorStores.stores) {
    retv += fStore->computeCost(makeComparableWithStored, stats, threadPool);
  }
  return retv;
}

int64_t Optimizer::variableBackupSize() const {
  int64_t size = 0;
  for (const auto& [_, vStore] : variableStores.stores) {
    size += vStore->totalSize();
  }
  return size;
}

void Optimizer::backupVariables(std::vector<double>& data) const {
  double* dataPtr = data.data();
  for (const auto& [_, vStore] : variableStores.stores) {
    dataPtr = vStore->backup(dataPtr);
  }
}

void Optimizer::restoreVariables(const std::vector<double>& data) {
  const double* dataPtr = data.data();
  for (const auto& [_, vStore] : variableStores.stores) {
    dataPtr = vStore->restore(dataPtr);
  }
}

std::tuple<double, double, double> Optimizer::applyStep(
    const Eigen::VectorXd& step,
    const BaSpaCho::PermutedCoalescedAccessor& acc) {
  double totMaxR = 0.0, totRSqSum = 0.0, totRSum = 0.0;
  int64_t totNVars = 0;
  for (const auto& [_, vStore] : variableStores.stores) {
    auto [maxR, rSqSum, rSum, nVars] = vStore->applyStep(step, acc);
    totMaxR = std::max(maxR, totMaxR);
    totRSqSum += rSqSum;
    totRSum += rSum;
    totNVars += nVars;
  }
  return {totMaxR, std::sqrt(totRSqSum / totNVars), totRSum / totNVars};
}

void Optimizer::addDamping(
    Eigen::VectorXd& hess,
    const BaSpaCho::PermutedCoalescedAccessor& acc,
    double lambda) const {
  const int64_t numVariables = paramSizes.size();
  for (int64_t i = 0; i < numVariables; i++) {
    auto diag = acc.diagBlock(hess.data(), i).diagonal();
    diag *= (1.0 + lambda);
    diag.array() += lambda;
  }
}

std::string Optimizer::solverToString(SolverType solverType) {
  switch (solverType) {
    case Solver_Direct:
      return "direct";
    case Solver_PCG_Trivial:
      return "trivial";
    case Solver_PCG_Jacobi:
      return "jacobi";
    case Solver_PCG_GaussSeidel:
      return "gauss-seidel";
    case Solver_PCG_LowerPrecSolve:
      return "lower-prec-solve";
    default:
      return "<unknown>";
  }
}

// creates a solver
BaSpaCho::SolverPtr Optimizer::initSolver(
    int numThreads,
    bool fullElim,
    const std::unordered_set<int64_t>& paramsToReorderLast) {
  // make sure all variables are registered
  registerAllVariables();

  // collect variable sizes and (lower) off-diagonal blocks that need to be set
  std::unordered_set<std::pair<int64_t, int64_t>, pair_hash> blockSet;
  for (const auto& [_, fStore] : factorStores.stores) {
    fStore->registerBlocks(blockSet);
  }
  std::vector<std::pair<int64_t, int64_t>> blocks(blockSet.begin(), blockSet.end());
  std::sort(blocks.begin(), blocks.end());
  blockSet.clear();

  // create a CSR structure for the parameter blocks (=compressed sparse rows)
  std::vector<int64_t> ptrs{0}, inds;
  int64_t curRow = 0;
  for (auto [row, col] : blocks) {
    while (curRow < row) {
      inds.push_back(curRow); // diagonal
      ptrs.push_back(inds.size());
      curRow++;
    }
    inds.push_back(col);
  }
  while (curRow < paramSizes.size()) {
    inds.push_back(curRow); // diagonal
    ptrs.push_back(inds.size());
    curRow++;
  }

  // create sparse linear solver
  return createSolver(
      {.numThreads = numThreads,
       .addFillPolicy = (fullElim ? BaSpaCho::AddFillComplete : BaSpaCho::AddFillForGivenElims)},
      paramSizes,
      BaSpaCho::SparseStructure(std::move(ptrs), std::move(inds)),
      elimRanges,
      paramsToReorderLast);
}

// creates a "solve" function that will either
// 1. invoke the direct solver
// 2. apply partial elimination, run PCG, backtrack to a full solution
std::pair<std::function<std::string()>, std::function<std::string(Eigen::VectorXd&)>>
Optimizer::factorSolveFunctions(
    BaSpaCho::Solver& solver,
    Eigen::VectorXd& hess,
    const Settings& settings,
    int iterativeStart) {
  if (settings.solverType == Solver_Direct) {
    return {
        [&]() -> std::string {
          TimePoint start = Clock::now();
          solver.factor(hess.data());
          TimePoint end = Clock::now();
          return "direct, factor: " + timeString(end - start) + ", ";
        },
        [&](Eigen::VectorXd& vec) -> std::string {
          TimePoint start = Clock::now();
          solver.solve(hess.data(), vec.data(), solver.order(), /* nRHS = */ 1);
          TimePoint end = Clock::now();
          return "solve: " + timeString(end - start);
        }};
  } else {
    std::shared_ptr<Preconditioner<double>> precond;
    if (settings.solverType == Solver_PCG_Trivial) {
      precond = std::make_shared<IdentityPrecond<double>>(solver, iterativeStart);
    } else if (settings.solverType == Solver_PCG_Jacobi) {
      precond = std::make_shared<BlockJacobiPrecond<double>>(solver, iterativeStart);
    } else if (settings.solverType == Solver_PCG_GaussSeidel) {
      precond = std::make_shared<BlockGaussSeidelPrecond<double>>(solver, iterativeStart);
    } else if (settings.solverType == Solver_PCG_LowerPrecSolve) {
      precond = std::make_shared<LowerPrecSolvePrecond<double>>(solver, iterativeStart);
    } else {
      throw std::runtime_error(
          "Unknown preconditioner " + std::to_string((int)settings.solverType));
    }

    int64_t order = solver.order();
    int64_t secStart = solver.spanVectorOffset(iterativeStart);
    int64_t secSize = order - secStart;
    struct PCGStats {
      void reset() {
        nPrecond = nMatVec = 0;
        totPrecondTime = totMatVecTime = 0.0;
      }
      int nPrecond = 0;
      double totPrecondTime = 0.0;
      int nMatVec = 0;
      double totMatVecTime = 0.0;
    };
    auto pcgStats = std::make_shared<PCGStats>();
    auto pcg = std::make_shared<PCG>(
        [=](Eigen::VectorXd& u, const Eigen::VectorXd& v) {
          TimePoint start = Clock::now();
          u.resize(v.size());
          (*precond)(u.data(), v.data());
          pcgStats->nPrecond++;
          pcgStats->totPrecondTime += std::chrono::duration<double>(Clock::now() - start).count();
        },
        [=, &solver, &hess](Eigen::VectorXd& u, const Eigen::VectorXd& v) {
          TimePoint start = Clock::now();
          u.resize(v.size());
          u.setZero();
          solver.addMvFrom(
              hess.data(),
              iterativeStart,
              v.data() - secStart,
              order,
              u.data() - secStart,
              order,
              1);
          pcgStats->nMatVec++;
          pcgStats->totMatVecTime += std::chrono::duration<double>(Clock::now() - start).count();
        },
        settings.pcgDesiredResidual,
        settings.pcgMaxIterations,
        /* verbose = */ false);

    return {
        [=, &solver, &hess]() -> std::string {
          TimePoint start = Clock::now();

          solver.factorUpTo(hess.data(), iterativeStart);
          TimePoint end_elim = Clock::now();

          precond->init(hess.data());
          TimePoint end_precond = Clock::now();

          std::stringstream ss;
          ss << "semi-precond(" << solverToString(settings.solverType)
             << "), factor[elims]: " << timeString(end_elim - start)
             << ", init precond: " << timeString(end_precond - end_elim);
          return ss.str();
        },
        [=, &solver, &hess](Eigen::VectorXd& vec) -> std::string {
          TimePoint start = Clock::now();

          solver.solveLUpTo(hess.data(), iterativeStart, vec.data(), order, /* nRHS = */ 1);
          TimePoint end_elim_solve = Clock::now();

          pcgStats->reset();
          Eigen::VectorXd tmp;
          auto [its, res] = pcg->solve(tmp, vec.segment(secStart, secSize));
          vec.segment(secStart, secSize) = tmp;
          TimePoint end_res_solve = Clock::now();

          solver.solveLtUpTo(hess.data(), iterativeStart, vec.data(), order, /* nRHS = */ 1);
          TimePoint end_backsubst = Clock::now();

          std::stringstream ss;
          ss << "\nsolve[elims]: " << timeString(end_elim_solve - start) //
             << ", reduced-PCG: " << timeString(end_res_solve - end_elim_solve) //
             << ", backsubst: " << timeString(end_backsubst - end_res_solve)
             << "\n(precond: " << pcgStats->nPrecond << "x"
             << microsecondsString(pcgStats->totPrecondTime * 1e6 / pcgStats->nPrecond)
             << ", matvec: " << pcgStats->nMatVec << "x"
             << microsecondsString(pcgStats->totMatVecTime * 1e6 / pcgStats->nMatVec)
             << ", iters: " << its << ", residual: " << res << ")";
          return ss.str();
        },
    };
  }
}

void Optimizer::initDirectSolverData(
    DirectSolverData& ds,
    const Settings& settings,
    const std::unordered_set<int64_t>& paramsToReorderLast) {
  ds.solver = initSolver(settings.numThreads, true, paramsToReorderLast);
  ds.grad.setZero(ds.solver->order());
  ds.hess.setZero(ds.solver->dataSize());

  dispenso::ThreadPool threadPool{settings.numThreads > 1 ? settings.numThreads : 0};
  ds.cost = computeGradHess(
      ds.grad.data(),
      ds.solver->accessor(),
      ds.hess.data(),
      /* updateCachedResults = */ false,
      /* dontRetryFailed = */ false,
      settings.numThreads > 1 ? &threadPool : nullptr);

  if (settings.damping > 0) {
    addDamping(ds.hess, ds.solver->accessor(), settings.damping);
  }
}

std::pair<Eigen::MatrixXd, std::vector<int64_t>> Optimizer::sparseElimMarginalInformation() {
  return sparseElimMarginalInformation({});
}

std::pair<Eigen::MatrixXd, std::vector<int64_t>> Optimizer::sparseElimMarginalInformation(
    const Settings& settings) {
  DirectSolverData ds;
  initDirectSolverData(ds, settings);

  const int64_t elimParamEnd = elimRanges.empty() ? 0 : elimRanges.back();
  ds.solver->factorUpTo(ds.hess.data(), elimParamEnd);

  // densified bottom right hessian block
  Eigen::MatrixXd denseHessian;
  ds.solver->skel().densify(denseHessian, ds.hess.data(), /* fillUpperHalf = */ true, elimParamEnd);

  // create list of indices
  std::vector<int64_t> varIndices(paramSizes.size() - elimParamEnd);
  for (int64_t varIndex = elimParamEnd; varIndex < (int64_t)paramSizes.size(); varIndex++) {
    varIndices[ds.solver->paramToSpan()[varIndex] - elimParamEnd] = varIndex;
  }

  return {denseHessian, varIndices};
}

void Optimizer::updateUnderConditioning(
    const std::unordered_map<int64_t, Eigen::VectorXd>& update) {
  updateUnderConditioning({}, update);
}

void Optimizer::updateUnderConditioning(
    const Settings& settings,
    const std::unordered_map<int64_t, Eigen::VectorXd>& update) {
  std::unordered_set<int64_t> indicesToOrderLast;
  for (const auto& [i, upd] : update) {
    BASPACHO_CHECK_GE(i, 0);
    indicesToOrderLast.insert(i);
  }

  DirectSolverData ds;
  initDirectSolverData(ds, settings, indicesToOrderLast);

  const auto& skel = ds.solver->skel();
  const int64_t minSpanIndex = skel.numSpans() - indicesToOrderLast.size();

  Eigen::VectorXd step = Eigen::VectorXd::Zero(ds.solver->order());
  int64_t startCond = skel.spanStart[minSpanIndex];
  const auto& accessor = ds.solver->accessor();
  for (const auto& [i, upd] : update) {
    int64_t start = accessor.paramStart(i);
    BASPACHO_CHECK_GE(start, startCond);
    int64_t size = accessor.paramSize(i);
    BASPACHO_CHECK_EQ(size, upd.size());
    step.segment(start, size) = upd;
  }

  // partial cholesky factor on columns up to `minSpanIndex`,
  // remaining borrom-right block becomes the (permuted) marginal information matrix
  ds.solver->factorUpTo(ds.hess.data(), minSpanIndex);

  // only apply the back substitution
  ds.solver->solveLtUpTo(ds.hess.data(), minSpanIndex, step.data(), step.size(), /* nRHS = */ 1);

  applyStep(step, accessor);
}

Optimizer::MarginalProblem Optimizer::computeMarginalProblem(const std::vector<int64_t>& indices) {
  return computeMarginalProblem({}, indices);
}

// returns {matrix, offsets}
Optimizer::MarginalProblem Optimizer::computeMarginalProblem(
    const Settings& settings,
    const std::vector<int64_t>& indices) {
  // taking index from UNREGISTERED variable?
  BASPACHO_CHECK(std::all_of(indices.begin(), indices.end(), [&](int64_t i) { return i >= 0; }));

  std::unordered_set<int64_t> indicesToOrderLast(indices.begin(), indices.end());
  BASPACHO_CHECK_EQ(indicesToOrderLast.size(), indices.size()); // expect no duplicates

  DirectSolverData ds;
  initDirectSolverData(ds, settings, indicesToOrderLast);

  // skel is the sparse structure of the factor, see baspacho/CoalescedBlockMatrix.h
  // here we just use the size/offsets of the `span` (ie reordered param blocks)
  const auto& skel = ds.solver->skel();
  const int64_t minSpanIndex = skel.numSpans() - indices.size();
  const int64_t offStart = skel.spanStart[minSpanIndex];
  std::vector<int64_t> spanOffsets(indices.size());
  std::vector<int64_t> spanSize(indices.size());
  std::vector<int64_t> offsets(spanSize.size() + 1);
  offsets[0] = 0;

  // compute offsets of params specified in `indices` in the (unpermuted) marginal info matrix
  for (size_t i = 0; i < indices.size(); i++) {
    const int64_t sIndex = ds.solver->paramToSpan()[indices[i]]; // un-permuted index in solver
    const int64_t sSize = skel.spanStart[sIndex + 1] - skel.spanStart[sIndex];
    BASPACHO_CHECK_GE(sIndex, minSpanIndex);
    spanOffsets[i] = skel.spanStart[sIndex];
    spanSize[i] = sSize;
    offsets[i + 1] = offsets[i] + sSize;
  }

  // partial cholesky factor on columns up to `minSpanIndex`,
  // remaining borrom-right block becomes the (permuted) marginal information matrix
  ds.solver->factorUpTo(ds.hess.data(), minSpanIndex);

  // densified the (permuted) bottom-right hessian block
  Eigen::MatrixXd H;
  ds.solver->skel().densify(H, ds.hess.data(), /* fillUpperHalf = */ true, minSpanIndex);

  // via partial solve, elimGrad becomes (whitened error + marginal error)
  Eigen::VectorXd elimGrad = ds.grad;
  ds.solver->solveLUpTo(
      ds.hess.data(), minSpanIndex, elimGrad.data(), elimGrad.size(), /* nRHS = */ 1);
  double correctedCost = ds.cost - elimGrad.head(offStart).squaredNorm() * 0.5;

  // fill error state for marginal problem (ie b)
  Eigen::VectorXd straightGrad(H.rows());

  // un-apply the permutation, so blocks are ordered in the same order as `indices` argument
  Eigen::MatrixXd straightH(H.rows(), H.cols());
  for (size_t i = 0; i < indices.size(); i++) {
    for (size_t j = 0; j < indices.size(); j++) {
      straightH.block(offsets[j], offsets[i], spanSize[j], spanSize[i]) =
          H.block(spanOffsets[j] - offStart, spanOffsets[i] - offStart, spanSize[j], spanSize[i]);
    }

    // no `- offStart` because `elimGrad` is the full gradient, not just marginal part
    straightGrad.segment(offsets[i], spanSize[i]) = elimGrad.segment(spanOffsets[i], spanSize[i]);
  }

  return {
      .H = straightH,
      .b = straightGrad,
      .cost = correctedCost,
      .offsets = offsets,
  };
}

std::vector<std::pair<Eigen::MatrixXd, std::vector<int64_t>>> Optimizer::computeJointCovariances(
    const std::vector<std::vector<int64_t>>& indicesLists) {
  return computeJointCovariances({}, indicesLists);
}

// returns a list of pairs {matrix, offsets}, which joint covariance and offsets for each
// list or parameters `indices` in the list `indicesList`
std::vector<std::pair<Eigen::MatrixXd, std::vector<int64_t>>> Optimizer::computeJointCovariances(
    const Settings& settings_,
    const std::vector<std::vector<int64_t>>& indicesLists) {
  for (const auto& indices : indicesLists) {
    // taking index from UNREGISTERED variable?
    BASPACHO_CHECK(std::all_of(indices.begin(), indices.end(), [&](int64_t i) { return i >= 0; }));

    // DUPLICATED indices? - this is probably not what you want
    BASPACHO_CHECK_EQ(std::unordered_set(indices.begin(), indices.end()).size(), indices.size());
  }

  DirectSolverData ds;
  Settings settings = settings_;
  while (true) {
    initDirectSolverData(ds, settings);
    ds.solver->factor(ds.hess.data());
    if (std::isfinite(ds.hess.lpNorm<1>())) {
      break;
    }
    if (settings.damping < 1e-9) {
      settings.damping += 1e-9;
    } else {
      settings.damping *= 2.0;
    }
    if (settings.log) {
      std::stringstream ss;
      ss << "unable to compute covariances, retrying with damping = " << settings.damping;
      settings.log(ss.str());
    }
  }

  Eigen::MatrixXd mat; // temporary for in-place solver
  std::vector<int64_t> spanIndices;
  std::vector<int64_t> spanSize;
  std::vector<int64_t> offsets;

  // return value
  std::vector<std::pair<Eigen::MatrixXd, std::vector<int64_t>>> covAndOffsets;

  for (size_t q = 0; q < indicesLists.size(); q++) {
    if (settings.log && (q % 250) == 0) {
      std::stringstream ss;
      ss << "Computing joint-block covariances: " << q << "/" << indicesLists.size();
      settings.log(ss.str());
    }
    const auto& indices = indicesLists[q];

    int64_t minSpanIndex = std::numeric_limits<int64_t>::max();
    spanIndices.resize(indices.size());
    spanSize.resize(indices.size());
    offsets.resize(indices.size() + 1);
    offsets[0] = 0;

    // compute offsets of params specified in `indices` in the (unpermuted) marginal info matrix
    const auto& skel = ds.solver->skel();
    for (size_t i = 0; i < indices.size(); i++) {
      const int64_t sIndex = ds.solver->paramToSpan()[indices[i]]; // un-permuted solver index
      const int64_t sSize = skel.spanStart[sIndex + 1] - skel.spanStart[sIndex];
      spanIndices[i] = sIndex;
      minSpanIndex = std::min(minSpanIndex, sIndex);
      spanSize[i] = sSize;
      offsets[i + 1] = offsets[i] + sSize;
    }

    // square block that will hold joint covariance for params in `indices`
    const int64_t covSize = offsets.back();
    Eigen::MatrixXd cov(covSize, covSize);

    // We can invoke a partial `solveL(t)From`, in-place solve on a bottom-right block. However
    // this this cannot be done from any param because columns of blocks are `coalesced` in the
    // supernodal structure, so align to the best possible `lump` (i.e. supernode)
    const int64_t sAtLumpStart = skel.lumpToSpan[skel.spanToLump[minSpanIndex]];
    const int64_t lumpsOffset = ds.solver->spanVectorOffset(sAtLumpStart);
    const int64_t matRows = ds.solver->order() - lumpsOffset;

    // call in-place solve on a set of columns corresponding to a param in `indices`
    for (size_t i = 0; i < indices.size(); i++) {
      const int64_t matCols = spanSize[i];
      const int64_t spanOffsetI = ds.solver->spanVectorOffset(spanIndices[i]);
      mat.setZero(matRows, matCols);
      mat.block(spanOffsetI - lumpsOffset, 0, matCols, matCols).setIdentity();

      ds.solver->solveLFrom(
          ds.hess.data(),
          sAtLumpStart,
          mat.data() - lumpsOffset, // pointer to vector only allocated from `lumpsOffset`
          matRows, // stride
          matCols); // nRHS
      ds.solver->solveLtFrom(
          ds.hess.data(),
          sAtLumpStart,
          mat.data() - lumpsOffset, // pointer to vector only allocated from `lumpsOffset`
          matRows, // stride
          matCols); // nRHS

      // copy the right blocks in the output covariance matrix
      for (size_t j = 0; j < indices.size(); j++) {
        const int64_t spanOffsetJ = ds.solver->spanVectorOffset(spanIndices[j]);
        cov.block(offsets[j], offsets[i], spanSize[j], matCols) =
            mat.block(spanOffsetJ - lumpsOffset, 0, spanSize[j], matCols);
      }
    }

    // add covariance matrix and offsets to the list of return values
    covAndOffsets.emplace_back(cov, offsets);
  }

  return covAndOffsets;
}

std::vector<Eigen::MatrixXd> Optimizer::computeCovariances(const std::vector<int64_t>& indices) {
  return computeCovariances({}, indices);
}

std::vector<Eigen::MatrixXd> Optimizer::computeCovariances(
    const Settings& settings_,
    const std::vector<int64_t>& indices) {
  // taking index from UNREGISTERED variable?
  BASPACHO_CHECK(std::all_of(indices.begin(), indices.end(), [&](int64_t i) { return i >= 0; }));

  // DUPLICATED indices? - this is probably not what you want
  BASPACHO_CHECK_EQ(std::unordered_set(indices.begin(), indices.end()).size(), indices.size());

  DirectSolverData ds;
  Settings settings = settings_;
  while (true) {
    initDirectSolverData(ds, settings);
    ds.solver->factor(ds.hess.data());
    if (std::isfinite(ds.hess.lpNorm<1>())) {
      break;
    }
    if (settings.damping < 1e-9) {
      settings.damping += 1e-9;
    } else {
      settings.damping *= 2.0;
    }
    if (settings.log) {
      std::stringstream ss;
      ss << "unable to compute covariances, retrying with damping = " << settings.damping;
      settings.log(ss.str());
    }
  }

  std::vector<Eigen::MatrixXd> covs;
  Eigen::MatrixXd mat; // temporary for in-place solver
  for (size_t i = 0; i < indices.size(); i++) {
    if (settings.log && (i % 250) == 0) {
      std::stringstream ss;
      ss << "Computing block covariances: " << i << "/" << indices.size();
      settings.log(ss.str());
    }

    const int64_t index = indices[i];
    const auto& skel = ds.solver->skel(); // plain (un-permuted) block structure
    const int64_t sIndex = ds.solver->paramToSpan()[index]; // un-permuted solver index
    const int64_t sSize = skel.spanStart[sIndex + 1] - skel.spanStart[sIndex];

    // we cannot do partial "solve from" from any span because they have been clustered together,
    // but only at "lump" boundaries. Therefore compute the lump-aligned span we can solve from.
    const int64_t offsetInLump = skel.spanOffsetInLump[sIndex];
    const int64_t sAtLumpStart = skel.lumpToSpan[skel.spanToLump[sIndex]];
    const int64_t lumpsOffset = ds.solver->spanVectorOffset(sAtLumpStart);
    const int64_t matRows = ds.solver->order() - lumpsOffset;
    const int64_t matCols = sSize;

    // build a matrix we can partially solve on (from sAtLumpStart), with identity block
    mat.setZero(matRows, matCols);
    mat.block(offsetInLump, 0, sSize, sSize).setIdentity();

    ds.solver->solveLFrom(
        ds.hess.data(),
        sAtLumpStart,
        mat.data() - lumpsOffset, // pointer to vector only allocated from `lumpsOffset`
        matRows, // stride
        matCols); // nRHS
    ds.solver->solveLtFrom(
        ds.hess.data(),
        sAtLumpStart,
        mat.data() - lumpsOffset, // pointer to vector only allocated from `lumpsOffset`
        matRows, // stride
        matCols); // nRHS

    // extract diagonal block from rectangle block of inverse Hessian
    covs.push_back(mat.block(offsetInLump, 0, sSize, sSize));
  }

  if (settings.log) {
    std::stringstream ss;
    ss << "Computing block covariances: done";
    settings.log(ss.str());
  }

  return covs;
}

Optimizer::Summary Optimizer::optimize() {
  return optimize({});
}

struct Optimizer::ExpectedValues {
  std::unordered_map<FactorStoreBase*, std::vector<std::tuple<double, double, double>>> s2xv;
  std::vector<std::pair<FactorStoreBase*, int>> debugFactors;
};

static void
computeDebugFactors(Optimizer::ExpectedValues& ev, bool stepApplied, const FactorDebugFunc& f) {
  for (auto [pFactorStore, i] : ev.debugFactors) {
    f(stepApplied, pFactorStore, i);
  }
}

void Optimizer::prepareExpectedValues(
    ExpectedValues& ev,
    const Eigen::VectorXd& step,
    const BaSpaCho::PermutedCoalescedAccessor& acc) const {
  for (const auto& [tid, factorStore] : factorStores.stores) {
    auto& xv = ev.s2xv[factorStore.get()];
    for (int64_t count = factorStore->numCosts(), i = 0; i < count; i++) {
      auto tup = factorStore->singleExpectedDelta(i, step.data(), acc);
      xv.push_back(tup);
    }
  }
}

void Optimizer::compareExpectedValues(ExpectedValues& ev) const {
  std::vector<std::tuple<double, double, FactorStoreBase*, int>> comparisons;
  for (const auto& [tid, factorStore] : factorStores.stores) {
    auto xvIt = ev.s2xv.find(factorStore.get());
    const auto& xv = xvIt->second;
    for (int64_t count = factorStore->numCosts(), i = 0; i < count; i++) {
      auto maybeV1 = factorStore->singleCost(i);
      auto [v0, delta, gradNorm] = xv[i];
      if (v0 < 0.0 || !maybeV1.has_value()) { // prev or next is failing?
        continue;
      }
      double v1 = maybeV1.value();
      double g = std::min(gradNorm, 1000.0);
      double score = std::abs(v0 + delta - v1) / (g + 1.0);
      comparisons.emplace_back(score, v1, factorStore.get(), i);
    }
  }

  std::sort(comparisons.begin(), comparisons.end(), [&](auto a, auto b) {
    return std::get<0>(a) < std::get<0>(b);
  });

  static constexpr int kNumFactors = 5;
  for (size_t q = std::max(0UL, comparisons.size() - kNumFactors); q < comparisons.size(); q++) {
    auto [score, v1, pFactorStore, i] = comparisons[q];
    auto xvIt = ev.s2xv.find(pFactorStore);
    const auto& xv = xvIt->second;
    auto [v0, delta, gradNorm] = xv[i];
    std::cout << "Factor: " << i << "@" << pFactorStore->name() << "\n" //
              << "Non-linearity score..: " << score << "\n" //
              << "Error value at x0....: " << v0 << "\n" //
              << "Predicted error at x1: " << v0 + delta << "\n" //
              << "Actual error at x1...: " << v1 << "\n" //
              << "Predicted delta......: " << delta << "\n" //
              << "Gradient norm........: " << gradNorm << "\n" //
              << std::endl;

    ev.debugFactors.emplace_back(pFactorStore, i);
  }
}

Optimizer::Summary Optimizer::optimize(const Settings& settings) {
  // create sparse linear solver
  BaSpaCho::SolverPtr solver = initSolver(
      settings.numThreads,
      (settings.solverType == Solver_Direct || settings.solverType == Solver_PCG_LowerPrecSolve));
  const auto& accessor = solver->accessor();

  // linear system data
  std::vector<double> variablesBackup(variableBackupSize());
  Eigen::VectorXd grad(solver->order());
  if (settings.log) {
    std::stringstream ss;
    ss << "allocating Hessian matrix: " << humanReadableSize(sizeof(double) * solver->dataSize());
    settings.log(ss.str());
  }
  Eigen::VectorXd hess(solver->dataSize());
  Eigen::VectorXd step(solver->order());
  auto [factorFunc, solveFunc] =
      factorSolveFunctions(*solver, hess, settings, elimRanges.empty() ? 0 : elimRanges.back());
  double damping = settings.damping;

  int iterationNum = 0;
  int lastImprovementIteration = 0; // last iteration we had a significant improvement
  int lastTroubledIteration = -10;
  double initialCost = 0.0, finalCost;
  double troubledSeqStartDamping = damping;
  int troubledSeqStart = 0, numTroubledSeqs = 0, largestTroubledSeq = 0;
  bool dontRetryFailed = false;

  dispenso::ThreadPool threadPool{settings.numThreads > 1 ? settings.numThreads : 0};

  // iteration loop
  while (true) {
    TimePoint start_it = Clock::now();

    if (settings.preStepCallback) {
      settings.preStepCallback(iterationNum);
    }

    grad.setZero();
    hess.setZero();
    double prevCost = computeGradHess(
        grad.data(),
        accessor,
        hess.data(),
        /* updateCachedResults = */ true,
        dontRetryFailed,
        settings.numThreads > 1 ? &threadPool : nullptr);
    TimePoint end_costs = Clock::now();

    finalCost = prevCost;
    if (iterationNum == 0) {
      initialCost = prevCost;
    }

    double modelCostReduction;
    std::string solverReport;
    do {
      addDamping(hess, accessor, damping);

      step = grad;
      solverReport = factorFunc();
      solverReport += solveFunc(step);

      // cost reduction that would occur perfectly quadratic
      modelCostReduction = step.dot(grad) * 0.5;

      if (modelCostReduction < 0) {
        grad.setZero();
        hess.setZero();
        computeGradHess(
            grad.data(),
            accessor,
            hess.data(),
            /* updateCachedResults = */ true,
            dontRetryFailed,
            settings.numThreads > 1 ? &threadPool : nullptr);

        damping *= settings.dampingAdjustOnFail;
        if (settings.log) {
          std::stringstream ss;
          ss << " ?:# quadratic model failing numerically, retrying... (damping: " << damping
             << ")";
          settings.log(ss.str());
        }
        continue;
      }
    } while (0);

    step *= -1.0;
    double gradNorm = grad.norm();
    double stepNorm = step.norm();

    // hook for inspection of most non-linear factors
    if (iterationNum + 1 == settings.triggerDebuggingOfNonlinearities) {
      if (settings.log) {
        settings.log("SmallThing: inspecting non-linearities...");
      }
      ExpectedValues ev;
      prepareExpectedValues(ev, step, accessor);

      backupVariables(variablesBackup);
      applyStep(step, accessor);

      compareExpectedValues(ev);

      if (settings.problematicFactorDebugFunc) {
        restoreVariables(variablesBackup);
        computeDebugFactors(ev, false, settings.problematicFactorDebugFunc);

        applyStep(step, accessor);
        computeDebugFactors(ev, false, settings.problematicFactorDebugFunc);
      }

      return {};
    }

    backupVariables(variablesBackup);
    auto [ratioS2VnInf, ratioS2Vn2, ratioS2Vn1] = applyStep(step, accessor);

    auto newFailureRateIsAcceptable = [&](const CostStats& stats) -> bool {
      const double newInvalidRate = stats.numInvalid / (stats.numTotal + 1.0);
      return newInvalidRate < 0.03 && (stats.numInvalid < stats.numPrevInvalid * 2.0 + 50);
    };

    CostStats stats;
    double newCost = computeCost(
        true, // makeComparableWithStored
        &stats,
        settings.numThreads > 1 ? &threadPool : nullptr);
    double costReduction = prevCost - newCost;
    double ratioReductionToCost = costReduction / newCost;
    double ratioReductionToExpected = costReduction / modelCostReduction;
    double appliedStepFactor = 1.0;
    bool failureRateIsAcceptable = newFailureRateIsAcceptable(stats);

    // if cost increase (or didn't decrease enough, compared with `modelCostReduction`),
    // we attempt by multiplying the step by a factor < 1.0. This factor is tentatively
    // obtained computing the gradient at the new estimate, interpolating model red
    if (settings.maxStepFactorAttempts > 0 &&
        (ratioReductionToExpected < settings.minRelativeCostReduction ||
         !failureRateIsAcceptable)) {
      Eigen::VectorXd gradNewX = Eigen::VectorXd::Zero(solver->order());
      computeGradHess(
          gradNewX.data(),
          accessor,
          nullptr,
          /* updateCachedResults = */ false,
          dontRetryFailed,
          settings.numThreads > 1 ? &threadPool : nullptr);
      double backRed = -gradNewX.dot(step) * 0.5; // model reduction when "going back"
      double stepFactor = backRed > 0 //
          ? (modelCostReduction / (modelCostReduction + backRed))
          : settings.stepFactorDecrease;

      for (int i = 0; i < settings.maxStepFactorAttempts; i++) {
        appliedStepFactor *= stepFactor;
        step *= stepFactor;
        restoreVariables(variablesBackup);
        applyStep(step, accessor);

        // check if we got to a better estimate
        CostStats statsAtF;
        double newCostAtF = computeCost(
            /* makeComparableWithStored = */ true,
            &statsAtF,
            settings.numThreads > 1 ? &threadPool : nullptr);
        double costReductionAtF = prevCost - newCost;
        double ratioReductionToExpectedAtF =
            costReductionAtF / (modelCostReduction * appliedStepFactor);
        bool failureRateIsAcceptableAtF = newFailureRateIsAcceptable(statsAtF);
        if (ratioReductionToExpectedAtF >= settings.minRelativeCostReduction &&
            failureRateIsAcceptableAtF) {
          newCost = newCostAtF;
          stats = statsAtF;
          costReduction = costReductionAtF;
          ratioReductionToExpected = ratioReductionToExpectedAtF;
          failureRateIsAcceptable = true;
          if (settings.log) {
            std::stringstream ss;
            ss << "\\!/ cost reduction obtained applying factor " << std::fixed
               << std::setprecision(2) << appliedStepFactor;
            settings.log(ss.str());
          }
          break;
        }

        // try sub-step: this is useful when moving in a narrow canyon with nonlinearities,
        // first step might increase the cost because we hit the canyon's walls, the substep
        // might correct this, and take us to an estimate where cost is lower
        if (settings.trySubStep) {
          // compute gradient
          gradNewX.setZero();
          computeGradHess(
              gradNewX.data(),
              accessor,
              nullptr,
              /* updateCachedResults = */ false,
              dontRetryFailed,
              settings.numThreads > 1 ? &threadPool : nullptr);

          // compute step (in place on gradient), we use existing factor and just call solve
          std::string subSolverReport = solveFunc(gradNewX);
          gradNewX *= -1.0;
          applyStep(gradNewX, accessor);

          // check if we got to a better estimate
          CostStats statsAtSub;
          double newCostAtSub = computeCost(
              /* makeComparableWithStored = */ true,
              &statsAtSub,
              settings.numThreads > 1 ? &threadPool : nullptr);
          double costReductionAtSub = prevCost - newCostAtSub;
          double ratioReductionToExpectedAtSub =
              costReductionAtSub / (modelCostReduction * appliedStepFactor);
          bool failureRateIsAcceptableAtSub = newFailureRateIsAcceptable(statsAtSub);
          if (ratioReductionToExpectedAtSub >= settings.minRelativeCostReduction &&
              failureRateIsAcceptableAtSub) {
            newCost = newCostAtSub;
            stats = statsAtSub;
            costReduction = costReductionAtSub;
            ratioReductionToExpected = ratioReductionToExpectedAtSub;
            failureRateIsAcceptable = true;
            if (settings.log) {
              std::stringstream ss;
              ss << "\\!/ cost reduction obtained applying factor " << std::fixed
                 << std::setprecision(2) << appliedStepFactor << " + sub-step:\n"
                 << indent(subSolverReport);
              settings.log(ss.str());
            }
            break;
          }
        }

        if (!dontRetryFailed) {
          dontRetryFailed = true;
          if (settings.log) {
            settings.log("\\!/ failing factors will no longer be retried!");
          }
        }

        stepFactor = settings.stepFactorDecrease;
      }
    }

    const char* smiley;
    const char* toleranceHit = //
        ratioReductionToCost < settings.relativeCostTolerance ? "relative cost"
        : costReduction < settings.absoluteCostTolerance      ? "absolute cost"
        : ratioS2Vn2 < settings.variablesTolerance            ? "variable"
                                                              : nullptr;
    if (newCost > prevCost || !failureRateIsAcceptable) { // failure!
      if (lastTroubledIteration != iterationNum - 1) { // start of new troubled sequence?
        troubledSeqStartDamping = damping;
        troubledSeqStart = iterationNum;
      }
      smiley = ":'(";
      damping *= settings.dampingAdjustOnFail;
      restoreVariables(variablesBackup);
      if (damping > settings.dampingMax) {
        if (settings.log) {
          std::stringstream ss;
          ss << "damping out of range, quadratic model failing?!";
          settings.log(ss.str());
        }
        break;
      }
      lastTroubledIteration = iterationNum;
    } else { // good, or average if less decrease than expected, or applied small fraction of step
      if (lastTroubledIteration == iterationNum - 1) { // end of troubled sequence?
        if (troubledSeqStartDamping < 1e1 && damping > 1e-3) {
          numTroubledSeqs++;
          largestTroubledSeq = std::max(largestTroubledSeq, iterationNum - troubledSeqStart);
        }
      }
      if (ratioReductionToExpected >= settings.minRelativeCostReduction &&
          appliedStepFactor > settings.minStepFactorForGood) {
        smiley = toleranceHit ? ";-|" : ":-)";
        damping *= settings.dampingAdjustOnGoodStep;
        damping = std::max(damping, settings.dampingMin);
      } else {
        smiley = ":-/";
        damping *= settings.dampingAdjustOnAverageStep;
      }
      finalCost = newCost;
    }

    TimePoint end = Clock::now();

    iterationNum++;
    if (settings.log) {
      std::stringstream ss;
      const double prevFailingRatio = stats.numPrevInvalid / std::max((double)stats.numTotal, 1.0);
      const double newFailingRatio = stats.numInvalid / std::max((double)stats.numTotal, 1.0);
      ss << " " << smiley << " cost: " << prevCost << " -> " << newCost << " ("
         << percentageString(newCost / prevCost - 1.0, 2) << "), t: " << timeString(end - start_it)
         << "\n" //
         << "     n." << iterationNum << "; g/H: " << timeString(end_costs - start_it)
         << ", solver:\n"
         << indent(solverReport) << "\n"
         << "     lmbd: " << damping << ", relRed: " << percentageString(ratioReductionToExpected)
         << ", improv: " << costReduction << ", modelImprov: " << modelCostReduction << "\n"
         << "    |G|: " << gradNorm << ", |S|: " << stepNorm << ", |s/v|_inf: " << ratioS2VnInf
         << ", |_2: " << ratioS2Vn2 << ", |_1: " << ratioS2Vn1 << "\n" //
         << "    Failing factors: " << prevFailingRatio << " ("
         << percentageString(prevFailingRatio) << ") -> " << newFailingRatio << " ("
         << percentageString(newFailingRatio) << ")";
      settings.log(ss.str());
    }
    if (!toleranceHit) {
      lastImprovementIteration = iterationNum;
    }
    if (iterationNum >= lastImprovementIteration + settings.stopIfNoImprovementFor &&
        iterationNum >= lastTroubledIteration + settings.distanceFromTroubledIteration) {
      if (settings.log) {
        std::stringstream ss;
        ss << " >_< converged! (hit " << toleranceHit << " tolerance, for "
           << settings.stopIfNoImprovementFor << " iterations)";
        settings.log(ss.str());
      }
      break;
    } else if (iterationNum >= settings.maxNumIterations) {
      if (settings.log) {
        std::stringstream ss;
        ss << " X-| iteration limit reached! (" << settings.maxNumIterations << " iterations)";
        settings.log(ss.str());
      }
      break;
    }
  }

  return {
      .initialCost = initialCost,
      .finalCost = finalCost,
      .numTroubledSeqs = numTroubledSeqs,
      .largestTroubledSeq = largestTroubledSeq,
      .numIterations = iterationNum,
  };
}

} // namespace small_thing
