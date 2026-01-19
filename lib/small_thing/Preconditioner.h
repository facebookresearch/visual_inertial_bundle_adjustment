/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iostream>
#include "baspacho/baspacho/Solver.h"

namespace small_thing {

/* abstract preconditioner class */
template <typename T>
class Preconditioner {
 public:
  using Vec = Eigen::Vector<T, Eigen::Dynamic>;

  virtual ~Preconditioner() {}

  virtual void init(T* data) = 0;

  virtual void operator()(T* outVec, const T* inVec) = 0;
};

/* dummy identity preconditioner */
template <typename T>
class IdentityPrecond : public Preconditioner<T> {
 public:
  /* `paramStart`: start of the system's bottom-right corner where we apply preconditioner */
  IdentityPrecond(const BaSpaCho::Solver& solver, int64_t paramStart)
      : solver(solver), vectorSize(solver.order() - solver.spanVectorOffset(paramStart)) {}

  virtual ~IdentityPrecond() override {}

  /* `matrixData`: the numeric factor data of the matrix we build a preconditioner for */
  virtual void init(T* /* matrixData */) override {}

  virtual void operator()(T* outVec, const T* inVec) override {
    Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(outVec, vectorSize) =
        Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>(inVec, vectorSize);
  }

 private:
  const BaSpaCho::Solver& solver;
  int64_t vectorSize;
};

/* Jacobi preconditioner, param-sized blocks are inverted */
template <typename T>
class BlockJacobiPrecond : public Preconditioner<T> {
 public:
  using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  /* `paramStart`: start of the system's bottom-right corner where we apply preconditioner */
  BlockJacobiPrecond(const BaSpaCho::Solver& solver, int64_t paramStart)
      : solver(solver),
        paramStart(paramStart),
        vectorSize(solver.order() - solver.spanVectorOffset(paramStart)) {
    int64_t numParams = solver.skel().numSpans();
    int64_t totSizeDiagBlocks = 0;
    for (int64_t p = paramStart; p < numParams; p++) {
      diagBlockOffset.push_back(totSizeDiagBlocks);
      int64_t paramSize = solver.spanVectorOffset(p + 1) - solver.spanVectorOffset(p);
      totSizeDiagBlocks += paramSize * paramSize;
    }
    diagBlockStorage.resize(totSizeDiagBlocks);
  }

  virtual ~BlockJacobiPrecond() override {}

  /* `matrixData`: the numeric factor data of the matrix we build a preconditioner for */
  virtual void init(T* matrixData) override {
    auto matrixBlockAccessor = solver.accessor();
    for (size_t i = 0; i < diagBlockOffset.size(); i++) {
      int64_t p = i + paramStart;
      int64_t paramSize = solver.spanVectorOffset(p + 1) - solver.spanVectorOffset(p);
      Eigen::Map<Mat> diagBlock(diagBlockStorage.data() + diagBlockOffset[i], paramSize, paramSize);
      diagBlock = matrixBlockAccessor.plainAcc.diagBlock(matrixData, p);
      Eigen::LLT<Eigen::Ref<Mat>> llt(diagBlock); // compute LLt in-place
    }
  }

  virtual void operator()(T* outVec, const T* inVec) override {
    Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> out(outVec, vectorSize);
    Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> in(inVec, vectorSize);
    out = in;

    int64_t vectorSectionStart = solver.spanVectorOffset(paramStart);
    for (size_t i = 0; i < diagBlockOffset.size(); i++) {
      int64_t paramIndex = i + paramStart;
      int64_t paramOffset = solver.spanVectorOffset(paramIndex);
      int64_t paramSize = solver.spanVectorOffset(paramIndex + 1) - paramOffset;
      int64_t paramRelativeOffset = paramOffset - vectorSectionStart;

      // solve, in-place, in the appropriate section of `out` vector
      Eigen::Map<Mat> diagBlock(diagBlockStorage.data() + diagBlockOffset[i], paramSize, paramSize);
      diagBlock.template triangularView<Eigen::Lower>().template solveInPlace<Eigen::OnTheLeft>(
          out.segment(paramRelativeOffset, paramSize));
      diagBlock.template triangularView<Eigen::Lower>()
          .transpose()
          .template solveInPlace<Eigen::OnTheLeft>(out.segment(paramRelativeOffset, paramSize));
    }
  }

 private:
  const BaSpaCho::Solver& solver;
  int64_t paramStart;
  int64_t vectorSize;
  std::vector<int64_t> diagBlockOffset;
  std::vector<T> diagBlockStorage;
};

/* Gauss-Seidel preconditioner, param-sized blocks are inverted */
template <typename T>
class BlockGaussSeidelPrecond : public Preconditioner<T> {
 public:
  /* `paramStart`: start of the system's bottom-right corner where we apply preconditioner */
  BlockGaussSeidelPrecond(const BaSpaCho::Solver& solver, int64_t paramStart)
      : solver(solver),
        paramStart(paramStart),
        vectorSize(solver.order() - solver.spanVectorOffset(paramStart)) {}

  virtual ~BlockGaussSeidelPrecond() override {}

  /* `matrixData`: the numeric factor data of the matrix we build a preconditioner for */
  virtual void init(T* matrixData) override {
    int64_t bottomRightOffset = solver.spanMatrixOffset(paramStart);
    bottomRightMatData.assign(matrixData + bottomRightOffset, matrixData + solver.dataSize());

    // call pseudo factor on "virtual" factor, ie where only bottom right is allocated.
    // guaranteed of pseudo factor is that only the bottom right corner is accessed.
    double* virtualFactorData = bottomRightMatData.data() - bottomRightOffset;
    solver.pseudoFactorFrom(virtualFactorData, paramStart);
  }

  virtual void operator()(T* outVec, const T* inVec) override {
    Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(outVec, vectorSize) =
        Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>(inVec, vectorSize);

    // call solve on "virtual" factor / and "virtual" vector, where only bottom right corner
    // of matrix and portion of the vector are allocated (no access outside allocated data)
    int64_t bottomRightOffset = solver.spanMatrixOffset(paramStart);
    const double* virtualFactorData = bottomRightMatData.data() - bottomRightOffset;

    int64_t vectorSectionStart = solver.spanVectorOffset(paramStart);
    double* virtualVectorData = outVec - vectorSectionStart;

    solver.solveLFrom(virtualFactorData, paramStart, virtualVectorData, vectorSize, 1);
    solver.solveLtFrom(virtualFactorData, paramStart, virtualVectorData, vectorSize, 1);
  }

 private:
  const BaSpaCho::Solver& solver;
  int64_t paramStart;
  int64_t vectorSize;
  std::vector<T> bottomRightMatData;
};

/* lower precision-solve preconditioner */
template <typename T>
class LowerPrecSolvePrecond;

template <>
class LowerPrecSolvePrecond<double> : public Preconditioner<double> {
 public:
  using T = double;

  /* `paramStart`: start of the system's bottom-right corner where we apply preconditioner */
  LowerPrecSolvePrecond(const BaSpaCho::Solver& solver, int64_t paramStart)
      : solver(solver),
        vectorSize(solver.order() - solver.spanVectorOffset(paramStart)),
        paramStart(paramStart) {}

  virtual ~LowerPrecSolvePrecond() override {}

  /* `matrixData`: the numeric factor data of the matrix we build a preconditioner for */
  virtual void init(double* matrixData) override {
    int64_t bottomRightOffset = solver.spanMatrixOffset(paramStart);
    int64_t bottomRightMatDataSize = solver.dataSize() - bottomRightOffset;
    bottomRightMatData.resize(bottomRightMatDataSize);
    float* virtualFactorData = bottomRightMatData.data() - bottomRightOffset;

    // Since we work in lower precision, factor might fail (and produce NaNs, if the matrix
    // is ill conditioned). If that's the case we increase progressively the values on the
    // diagonanal, till when it suceeds. In such a case what we get is the Cholesky factor not
    // of the original matrix, but of a matrix with increased diagonal to improve conditioning.
    // This is ok, as anyway we are using this not for a direct solve step but as a
    // preconditioner in the iterative PCG algoroithm.
    float epsilon = 0.0;
    do {
      // copy data from original matrix
      Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>(
          bottomRightMatData.data(), bottomRightMatDataSize) =
          Eigen::Map<const Eigen::Vector<double, Eigen::Dynamic>>(
              matrixData + bottomRightOffset, bottomRightMatDataSize)
              .cast<float>();

      // apply damping on diagonal (d -> d*(1+epsilon) + epsilon)
      if (epsilon > 0) {
        auto matrixBlockAccessor = solver.accessor();
        for (int64_t i = paramStart; i < solver.skel().numSpans(); i++) {
          auto diagBlock = matrixBlockAccessor.plainAcc.diagBlock(virtualFactorData, i).diagonal();
          diagBlock.diagonal() *= 1.0 + epsilon;
          diagBlock.diagonal().array() += epsilon;
        }
        epsilon *= 3.0;
      } else {
        epsilon = 1e-8;
      }

      // call factor on the bottom-right corner of the "virtual" factor
      solver.factorFrom(virtualFactorData, paramStart);
    } while (!std::isfinite(Eigen::Map<Eigen::Vector<float, Eigen::Dynamic>>(
                                bottomRightMatData.data(), bottomRightMatDataSize)
                                .sum()));
  }

  virtual void operator()(double* outVec, const double* inVec) override {
    // copy, casting to float
    Eigen::Vector<float, Eigen::Dynamic> temp =
        Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>(inVec, vectorSize).cast<float>();

    // call solve on "virtual" factor / and "virtual" vector, where only bottom right corner
    // of matrix and portion of the vector are allocated (no access outside allocated data)
    int64_t bottomRightOffset = solver.spanMatrixOffset(paramStart);
    const float* virtualFactorData = bottomRightMatData.data() - bottomRightOffset;

    int64_t vectorSectionStart = solver.spanVectorOffset(paramStart);
    float* virtualVectorData = temp.data() - vectorSectionStart;

    solver.solveLFrom(virtualFactorData, paramStart, virtualVectorData, vectorSize, 1);
    solver.solveLtFrom(virtualFactorData, paramStart, virtualVectorData, vectorSize, 1);

    // cast back to double
    Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>(outVec, vectorSize) = temp.cast<double>();
  }

 private:
  const BaSpaCho::Solver& solver;
  int64_t vectorSize;
  int64_t paramStart;
  std::vector<float> bottomRightMatData;
};

} // namespace small_thing
