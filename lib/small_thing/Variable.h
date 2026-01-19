/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <small_thing/Common.h>

#include <baspacho/baspacho/Accessor.h>
#include <sophus/se2.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Geometry>

namespace small_thing {

// Interface allowing to work with give variable type. It contains sizes of
// backup data and tangent space, boxPlus, and utilities to backup restore
template <typename T>
struct VarSpec;

// specialization for statically sized Eigen::Vector
template <int N>
struct VarSpec<Eigen::Vector<double, N>> {
  static constexpr int DataDimSpec = N;
  static constexpr int TangentDimSpec = N;
  using DataType = Eigen::Vector<double, N>;

  // return an estimate of |step| / |variable|
  template <typename Derived>
  static double applyBoxPlus(DataType& value, const Eigen::MatrixBase<Derived>& step) {
    value += step;
    return step.template lpNorm<Eigen::Infinity>() /
        (1.0 + value.template lpNorm<Eigen::Infinity>());
  }

  template <typename Derived>
  static void
  boxMinus(const DataType& value, const DataType& base, Eigen::MatrixBase<Derived>& delta) {
    delta = value - base;
  }

  template <typename Derived>
  static void getData(const DataType& value, Eigen::MatrixBase<Derived>& data) {
    data = value;
  }

  template <typename Derived>
  static void setData(DataType& value, const Eigen::MatrixBase<Derived>& data) {
    value = data;
  }
};

template <>
struct VarSpec<Eigen::Vector<double, Eigen::Dynamic>> {
  static constexpr int DataDimSpec = Eigen::Dynamic;
  static constexpr int TangentDimSpec = Eigen::Dynamic;
  using DataType = Eigen::Vector<double, Eigen::Dynamic>;

  static int getDynamicDataDim(const DataType& value) {
    return value.size();
  }

  static int getDynamicTangentDim(const DataType& value) {
    return value.size();
  }

  // return an estimate of |step| / |variable|
  template <typename Derived>
  static double applyBoxPlus(DataType& value, const Eigen::MatrixBase<Derived>& step) {
    value += step;
    return step.template lpNorm<Eigen::Infinity>() /
        (1.0 + value.template lpNorm<Eigen::Infinity>());
  }

  template <typename Derived>
  static void
  boxMinus(const DataType& value, const DataType& base, Eigen::MatrixBase<Derived>& delta) {
    delta = value - base;
  }

  template <typename Derived>
  static void getData(const DataType& value, Eigen::MatrixBase<Derived>& data) {
    data = value;
  }

  template <typename Derived>
  static void setData(DataType& value, const Eigen::MatrixBase<Derived>& data) {
    value = data;
  }
};

// specialization for Sophus::SE3d
template <>
struct VarSpec<Sophus::SE3d> {
  static constexpr int DataDimSpec = 7;
  static constexpr int TangentDimSpec = 6;
  using DataType = Sophus::SE3d;

  // return an estimate of |step| / |variable|
  template <typename Derived>
  static double applyBoxPlus(DataType& value, const Eigen::MatrixBase<Derived>& step) {
    value = Sophus::SE3d::exp(step) * value;
    return std::max(
        step.template tail<3>().template lpNorm<Eigen::Infinity>(),
        step.template head<3>().template lpNorm<Eigen::Infinity>() /
            (1.0 + value.translation().template lpNorm<Eigen::Infinity>()));
  }

  template <typename Derived>
  static void
  boxMinus(const DataType& value, const DataType& base, Eigen::MatrixBase<Derived>& delta) {
    delta = (value * base.inverse()).log();
  }

  template <typename Derived>
  static void getData(const DataType& value, Eigen::MatrixBase<Derived>& data) {
    data = Eigen::Map<const Eigen::Vector<double, DataDimSpec>>(value.data());
  }

  template <typename Derived>
  static void setData(Sophus::SE3d& value, const Eigen::MatrixBase<Derived>& data) {
    Eigen::Map<Eigen::Vector<double, DataDimSpec>>(value.data()) = data;
  }
};

// specialization for Sophus::SE2d
template <>
struct VarSpec<Sophus::SE2d> {
  static constexpr int DataDimSpec = 4;
  static constexpr int TangentDimSpec = 3;
  using DataType = Sophus::SE2d;

  // return an estimate of |step| / |variable|
  template <typename Derived>
  static double applyBoxPlus(DataType& value, const Eigen::MatrixBase<Derived>& step) {
    value = Sophus::SE2d::exp(step) * value;
    return std::max(
        step.template tail<2>().template lpNorm<Eigen::Infinity>(),
        step.template head<2>().template lpNorm<Eigen::Infinity>() /
            (1.0 + value.translation().template lpNorm<Eigen::Infinity>()));
  }

  template <typename Derived>
  static void
  boxMinus(const DataType& value, const DataType& base, Eigen::MatrixBase<Derived>& delta) {
    delta = (value * base.inverse()).log();
  }

  template <typename Derived>
  static void getData(const DataType& value, Eigen::MatrixBase<Derived>& data) {
    data = Eigen::Map<const Eigen::Vector<double, DataDimSpec>>(value.data());
  }

  template <typename Derived>
  static void setData(Sophus::SE2d& value, const Eigen::MatrixBase<Derived>& data) {
    Eigen::Map<Eigen::Vector<double, DataDimSpec>>(value.data()) = data;
  }
};

// vector with prescribed norm
struct S2 {
  double radius;
  Eigen::Vector3d vec;

  inline static Eigen::Matrix<double, 2, 3> ortho(const Eigen::Vector3d& v) {
    Eigen::Vector3d t1 = Eigen::Vector3d::Zero();
    double xAbs = std::abs(v.x()), yAbs = std::abs(v.y()), zAbs = std::abs(v.z());
    int coord = xAbs < std::min(yAbs, zAbs) ? 0 : yAbs < zAbs ? 1 : 2;
    t1[coord] = 1.0;
    double vSqNorm = v.squaredNorm();
    double vNorm = std::sqrt(vSqNorm);

    Eigen::Matrix<double, 2, 3> res;
    res.row(0) = (t1 - (t1.dot(v) / vSqNorm) * v).normalized();
    res.row(1) = res.row(0).cross(v) / vNorm;
    return res;
  }

  friend S2 operator*(const Sophus::SO3d& so3, const S2& s2) {
    return {.radius = s2.radius, .vec = so3 * s2.vec};
  }
};

// specialization for vector with prescribed norm
template <>
struct VarSpec<S2> {
  static constexpr int DataDimSpec = 3;
  static constexpr int TangentDimSpec = 2;
  using DataType = S2;
  // return an estimate of |step| / |variable|
  template <typename Derived>
  static double applyBoxPlus(DataType& value, const Eigen::MatrixBase<Derived>& step) {
    double angle = step.norm() / value.radius;
    double factor = angle > 1e-4 ? (tan(angle) / angle) : (1.0 + angle * angle / 3.0);
    value.vec = (value.vec + S2::ortho(value.vec).transpose() * (factor * step)).normalized() *
        value.radius;
    return angle;
  }

  template <typename Derived>
  static void
  boxMinus(const DataType& value, const DataType& base, Eigen::MatrixBase<Derived>& delta) {
    Eigen::Vector3d dv = value.vec.normalized() - base.vec.normalized();
    double angle = 2.0 * asin(dv.norm() * 0.5);
    double factor = 1.0 / cos(angle);
    delta = factor * (S2::ortho(base.vec) * dv * value.radius);
  }

  template <typename Derived>
  static void getData(const DataType& value, Eigen::MatrixBase<Derived>& data) {
    data = value.vec;
  }

  template <typename Derived>
  static void setData(DataType& value, const Eigen::MatrixBase<Derived>& data) {
    value.vec = data;
  }
};

// Generic variable class
constexpr int64_t kUnsetIndex = -1;
constexpr int64_t kConstantVar = -2;

struct VarBase {
  // mark as constant
  void setConstant(bool constant = true) {
    // cannot tweak constness, if registered
    BASPACHO_CHECK(!isRegistered());
    index = constant ? kConstantVar : kUnsetIndex;
  }

  // check if it was set to constant
  bool isSetToConstant() const {
    return index == kConstantVar;
  }

  // return true if registered in the optimized
  bool isRegistered() const {
    return index >= 0;
  }

  // return true if still needs registration as non-const optimization variable
  bool isUnset() const {
    return index == kUnsetIndex;
  }

  int64_t index = kUnsetIndex;
};

template <typename DType>
struct Variable : public VarBase {
  using DataType = DType;
  static constexpr int DataDimSpec = VarSpec<DataType>::DataDimSpec;
  static constexpr int TangentDimSpec = VarSpec<DataType>::TangentDimSpec;

  inline int getDataDim() const {
    if constexpr (DataDimSpec != Eigen::Dynamic) {
      return DataDimSpec;
    } else {
      return VarSpec<DataType>::getDynamicDataDim(value);
    }
  }

  inline int getTangentDim() const {
    if constexpr (TangentDimSpec != Eigen::Dynamic) {
      return TangentDimSpec;
    } else {
      return VarSpec<DataType>::getDynamicTangentDim(value);
    }
  }

  template <typename... Args>
  Variable(Args&&... args) : value(std::forward<Args>(args)...) {}

  DataType value;
};

// "opaque" (ie non-templated) base class for typed variable databases classes
class VariableStoreBase {
 public:
  virtual ~VariableStoreBase() {}

  // return (max ratio, sum ratio^2, sum ratio^1, num vars)
  virtual std::tuple<double, double, double, int64_t> applyStep(
      const Eigen::VectorXd& step,
      const BaSpaCho::PermutedCoalescedAccessor& acc) const = 0;

  virtual int64_t totalSize() const = 0;

  virtual int64_t totalVarDimensionality() const = 0;

  virtual double* backup(double* data) const = 0;

  virtual const double* restore(const double* data) const = 0;

  virtual void unregisterAll() = 0;
};

// specific class storing variables, can be casted down to "opaque" non-templated VariableStoreBase
template <typename Variable>
class VariableStore : public VariableStoreBase {
  static_assert(std::is_base_of_v<VarBase, Variable>);

 public:
  virtual ~VariableStore() override {}

  virtual int64_t totalSize() const override {
    if constexpr (Variable::DataDimSpec != Eigen::Dynamic) {
      return Variable::DataDimSpec * variables.size();
    } else {
      int64_t sum = 0;
      for (const auto& var : variables) {
        sum += var->getDataDim();
      }
      return sum;
    }
  }

  virtual int64_t totalVarDimensionality() const override {
    int64_t sum = 0;
    for (const auto& var : variables) {
      if (!var->isSetToConstant()) {
        sum += var->getTangentDim();
      }
    }
    return sum;
  }

  virtual double* backup(double* data) const override {
    for (const auto& var : variables) {
      Eigen::Map<Eigen::Vector<double, Variable::DataDimSpec>> dataMap(data, var->getDataDim());
      VarSpec<typename Variable::DataType>::getData(var->value, dataMap);
      data += var->getDataDim();
    }
    return data;
  }

  virtual const double* restore(const double* data) const override {
    for (const auto& var : variables) {
      Eigen::Map<const Eigen::Vector<double, Variable::DataDimSpec>> dataMap(
          data, var->getDataDim());
      VarSpec<typename Variable::DataType>::setData(var->value, dataMap);
      data += var->getDataDim();
    }
    return data;
  }

  // return (max ratio, sum ratio^2, sum ratio^1, num vars)
  virtual std::tuple<double, double, double, int64_t> applyStep(
      const Eigen::VectorXd& step,
      const BaSpaCho::PermutedCoalescedAccessor& acc) const override {
    double maxR = 0.0, rSqSum = 0.0, rSum = 0.0;
    for (const auto& var : variables) {
      double s2v;
      if constexpr (Variable::TangentDimSpec != Eigen::Dynamic) {
        s2v = VarSpec<typename Variable::DataType>::applyBoxPlus(
            var->value, step.segment<Variable::TangentDimSpec>(acc.paramStart(var->index)));
      } else {
        s2v = VarSpec<typename Variable::DataType>::applyBoxPlus(
            var->value, step.segment(acc.paramStart(var->index), var->getTangentDim()));
      }
      maxR = std::max(s2v, maxR);
      rSqSum += s2v * s2v;
      rSum += s2v;
    }
    return {maxR, rSqSum, rSum, variables.size()};
  }

  virtual void unregisterAll() override {
    for (auto& v : variables) {
      v->index = kUnsetIndex;
    }
    variables.clear();
  }

  std::vector<Variable*> variables;
};

} // namespace small_thing
