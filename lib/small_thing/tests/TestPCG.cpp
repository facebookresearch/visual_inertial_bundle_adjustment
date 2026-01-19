/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include "baspacho/testing/TestingUtils.h"
#include "small_thing/PCG.h"
#include "small_thing/Preconditioner.h"

using namespace BaSpaCho;
using namespace ::BaSpaCho::testing_utils;
using namespace std;
using namespace small_thing;

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using Vector = Eigen::Vector<T, Eigen::Dynamic>;

void runPreconditionerTest(
    const std::string& precondArg,
    int expectedSolveBeforeIt,
    int seed = 37) {
  // create random problem
  int numParams = 215;
  auto colBlocks = randomCols(numParams, 0.03, 57 + seed);
  colBlocks = makeIndependentElimSet(colBlocks, 0, 150);
  SparseStructure ss = columnsToCscStruct(colBlocks).transpose();
  vector<int64_t> paramSize = randomVec(ss.ptrs.size() - 1, 2, 3, 47);

  // create solver, with independent elim set [0, 100]
  auto fillPolicy = (precondArg == "lower-prec-solve") ? AddFillComplete : AddFillForAutoElims;
  auto solver =
      createSolver({.backend = BackendRef, .addFillPolicy = fillPolicy}, paramSize, ss, {0, 100});

  // use internal sparse elim range limit
  int64_t nocross = solver->sparseEliminationRanges().back();

  int order = solver->order();
  BASPACHO_CHECK_EQ(solver->skel().spanOffsetInLump[nocross], 0);

  // matrix data
  using T = double;
  vector<T> matData = randomData<T>(solver->dataSize(), -1.0, 1.0, 9 + seed);

  // add some randomized damping
  {
    auto acc = solver->accessor();
    mt19937 gen(seed);
    uniform_real_distribution<> dis(order * 0.1, order * 0.5);
    for (int64_t i = 0; i < (int64_t)paramSize.size(); i++) {
      acc.plainAcc.diagBlock(matData.data(), i).diagonal().array() += dis(gen);
    }
  }

  // b (=RHS)
  vector<T> bData = randomData<T>(order * 1, -1.0, 1.0, 49 + seed);
  Vector<T> b = Eigen::Map<Vector<T>>(bData.data(), order, 1);
  Vector<T> x = b;

  // factor and solve up to "nocross" value
  vector<T> origMatData = matData;
  solver->factorUpTo(matData.data(), nocross);
  solver->solveLUpTo(matData.data(), nocross, x.data(), order, 1);

  // create preconditioner
  std::unique_ptr<Preconditioner<double>> precond;
  if (precondArg == "none") {
    precond.reset(new IdentityPrecond<double>(*solver, nocross));
  } else if (precondArg == "jacobi") {
    precond.reset(new BlockJacobiPrecond<double>(*solver, nocross));
  } else if (precondArg == "gauss-seidel") {
    precond.reset(new BlockGaussSeidelPrecond<double>(*solver, nocross));
  } else if (precondArg == "lower-prec-solve") {
    precond.reset(new LowerPrecSolvePrecond<double>(*solver, nocross));
  } else {
    std::cout << "no such preconditioner '" << precondArg << "'" << std::endl;
    return;
  }
  precond->init(matData.data());

  // set up PCG in elimination area
  int64_t secStart = solver->spanVectorOffset(nocross);
  int64_t secSize = order - secStart;
  PCG pcg(
      // ApplyInvM() =
      [&](Eigen::VectorXd& u, const Eigen::VectorXd& v) {
        u.resize(v.size());
        (*precond)(u.data(), v.data());
      },
      // ApplyA() =
      [&](Eigen::VectorXd& u, const Eigen::VectorXd& v) {
        u.resize(v.size());
        u.setZero();
        solver->addMvFrom(
            matData.data(), nocross, v.data() - secStart, order, u.data() - secStart, order, 1);
      },
      3e-10,
      40,
      false);

  // run PCG on reduced system
  Eigen::VectorXd tmp;
  auto pcsResult = pcg.solve(tmp, x.segment(secStart, secSize));
  x.segment(secStart, secSize) = tmp;

  ASSERT_LT(pcsResult.numIterations, expectedSolveBeforeIt);
  ASSERT_NEAR(pcsResult.relativeResidual, 0, 1e-9);

  // trace-back
  solver->solveLtUpTo(matData.data(), nocross, x.data(), order, 1);

  // compute b2 = Ax on the full system
  Vector<T> b2(order);
  b2.setZero();
  solver->addMvFrom(origMatData.data(), 0, x.data(), order, b2.data(), order, 1);

  // verify residual
  double verifyRelResidual = (b - b2).norm() / b.norm();
  ASSERT_NEAR(verifyRelResidual, 0, 1e-9);
}

TEST(TestPrecond, Identity) {
  runPreconditionerTest("none", 30);
}

TEST(TestPrecond, Jacobi) {
  runPreconditionerTest("jacobi", 12);
}

TEST(TestPrecond, GaussSeidel) {
  runPreconditionerTest("gauss-seidel", 6);
}

TEST(TestPrecond, LowerPrecSolve) {
  runPreconditionerTest("lower-prec-solve", 5);
}
