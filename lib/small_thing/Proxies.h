/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <small_thing/CondensedFactor.h>

namespace small_thing {

template <typename T>
using StdVector = std::vector<T>;

template <typename T>
using StdArray1 = std::array<T, 1>;

// Special case of no transformation but just forwarding variables.
// Notice that in this case variables can be dynamically sized (eg Eigen::VectorXd),
// while the more complex proxy types do not allow this.
template <typename Data, template <class> class Container>
struct DirectForwardGen : DirectForwardBase {
  using Var = Variable<Data>;

  DirectForwardGen(Container<Var*>&& vars, Container<Data>&& linpts)
      : vars(std::move(vars)), linpts(std::move(linpts)) {
    BASPACHO_CHECK_EQ(vars.size(), linpts.size());
  }

  int64_t numVariables() const {
    return vars.size();
  }

  Var& getVar(int64_t i) const {
    return *vars[i];
  }

  // this must be functional even if the variable is set to constant
  template <typename Derived>
  void eval(int64_t i, Eigen::MatrixBase<Derived>&& delta) const {
    VarSpec<Data>::boxMinus(vars[i]->value, linpts[i], delta);
  }

  Container<Var*> vars;
  Container<Data> linpts;
};

template <typename Data>
using DirectForwards = DirectForwardGen<Data, StdVector>;
template <typename Data>
using DirectForward = DirectForwardGen<Data, StdArray1>;

// example proxy, return relative vector value
template <template <class> class Container>
struct ProxyS2Gen : ProxyBase {
  using InputVariablesType = std::tuple<Variable<Sophus::SE3d>*, Variable<S2>*>;
  static constexpr int ResultDim = 2;
  using ResultType = Eigen::Vector<double, ResultDim>;
  using JacobianType = std::tuple<Eigen::Matrix<double, 2, 6>, Eigen::Matrix<double, 2, 2>>;

  ProxyS2Gen(
      Variable<Sophus::SE3d>* T_loc_w,
      const Container<Variable<S2>*>& s2_w,
      const Container<S2>& linPt_s2_w)
      : T_loc_w(T_loc_w), s2_w(s2_w), linPt_s2_w(linPt_s2_w) {
    BASPACHO_CHECK_EQ(s2_w.size(), linPt_s2_w.size());
  }

  int64_t numEntries() const {
    return s2_w.size();
  }

  InputVariablesType inputVariables(int64_t i) const {
    return {T_loc_w, s2_w[i]};
  }

  ResultType eval(int64_t i) const {
    ResultType delta;
    VarSpec<S2>::boxMinus(T_loc_w->value.so3() * s2_w[i]->value, linPt_s2_w[i], delta);
    return delta;
  }

  ResultType eval(int64_t i, JacobianType* jac) const {
    const S2 transformed_var = T_loc_w->value.so3() * s2_w[i]->value;
    ResultType delta;
    VarSpec<S2>::boxMinus(transformed_var, linPt_s2_w[i], delta);

    std::get<0>(*jac) << Eigen::Matrix<double, 2, 3>::Zero(),
        S2::ortho(linPt_s2_w[i].vec) * Sophus::SO3d::hat(-transformed_var.vec);
    std::get<1>(*jac) = S2::ortho(linPt_s2_w[i].vec) * T_loc_w->value.so3().matrix() *
        S2::ortho(s2_w[i]->value.vec).transpose();
    return delta;
  }

  Variable<Sophus::SE3d>* T_loc_w;
  Container<Variable<S2>*> s2_w;
  Container<S2> linPt_s2_w;
};

using ProxyS2s = ProxyS2Gen<StdVector>;
using ProxyS2 = ProxyS2Gen<StdArray1>;

// example proxy, return relative vector value
template <int N, template <class> class Container>
struct ProxyRelativeVecsGen : ProxyBase {
  using InputVariablesType =
      std::tuple<Variable<Eigen::Vector<double, N>>*, Variable<Eigen::Vector<double, N>>*>;
  static constexpr int ResultDim = N;
  using ResultType = Eigen::Vector<double, ResultDim>;
  using JacobianType = std::tuple<Eigen::Matrix<double, N, N>, Eigen::Matrix<double, N, N>>;

  ProxyRelativeVecsGen(
      Variable<Eigen::Vector<double, N>>* T_p0_w,
      const Container<Variable<Eigen::Vector<double, N>>*>& T_pi_w,
      const Container<Eigen::Vector<double, N>>& linPt_T_pi_p0)
      : T_p0_w(T_p0_w), T_pi_w(T_pi_w), linPt_T_pi_p0(linPt_T_pi_p0) {
    BASPACHO_CHECK_EQ(T_pi_w.size(), linPt_T_pi_p0.size());
  }

  int64_t numEntries() const {
    return T_pi_w.size();
  }

  InputVariablesType inputVariables(int64_t i) const {
    return {T_p0_w, T_pi_w[i]};
  }

  ResultType eval(int64_t i) const {
    return T_pi_w[i]->value - T_p0_w->value - linPt_T_pi_p0[i];
  }

  ResultType eval(int64_t i, JacobianType* jac) const {
    std::get<0>(*jac).setIdentity();
    std::get<0>(*jac) *= -1;
    std::get<1>(*jac).setIdentity();
    return T_pi_w[i]->value - T_p0_w->value - linPt_T_pi_p0[i];
  }

  Variable<Eigen::Vector<double, N>>* T_p0_w;
  Container<Variable<Eigen::Vector<double, N>>*> T_pi_w;
  Container<Eigen::Vector<double, N>> linPt_T_pi_p0;
};

template <int N>
using ProxyRelativeVecs = ProxyRelativeVecsGen<N, StdVector>;
template <int N>
using ProxyRelativeVec = ProxyRelativeVecsGen<N, StdArray1>;

// example proxy: return relative poses, compute offset of rel pose from linearization points
template <template <class> class Container>
struct ProxyRelativePosesGen : ProxyBase {
  using InputVariablesType = std::tuple<Variable<Sophus::SE3d>*, Variable<Sophus::SE3d>*>;
  static constexpr int ResultDim = 6;
  using ResultType = Eigen::Vector<double, ResultDim>;
  using JacobianType = std::tuple<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 6>>;

  ProxyRelativePosesGen(
      Variable<Sophus::SE3d>* T_p0_w,
      const Container<Variable<Sophus::SE3d>*>& T_pi_w,
      const Container<Sophus::SE3d>& linPt_T_pi_p0)
      : T_p0_w(T_p0_w), T_pi_w(T_pi_w), linPt_T_pi_p0(linPt_T_pi_p0) {
    BASPACHO_CHECK_EQ(T_pi_w.size(), linPt_T_pi_p0.size());
  }

  int64_t numEntries() const {
    return T_pi_w.size();
  }

  InputVariablesType inputVariables(int64_t i) const {
    return {T_p0_w, T_pi_w[i]};
  }

  ResultType eval(int64_t i) const {
    return (T_pi_w[i]->value * T_p0_w->value.inverse() * linPt_T_pi_p0[i].inverse()).log();
  }

  ResultType eval(int64_t i, JacobianType* jac) const {
    Sophus::SE3d T_pi_p0 = T_pi_w[i]->value * T_p0_w->value.inverse();
    ResultType logDelta = (T_pi_p0 * linPt_T_pi_p0[i].inverse()).log();

    const Eigen::Matrix<double, 6, 6> dLogDelta_dLeftPi =
        Sophus::SE3d::leftJacobianInverse(logDelta);
    std::get<0>(*jac) = -dLogDelta_dLeftPi * T_pi_p0.Adj();
    std::get<1>(*jac) = dLogDelta_dLeftPi;
    return logDelta;
  }

  Variable<Sophus::SE3d>* T_p0_w;
  Container<Variable<Sophus::SE3d>*> T_pi_w;
  Container<Sophus::SE3d> linPt_T_pi_p0;
};

using ProxyRelativePoses = ProxyRelativePosesGen<StdVector>;
using ProxyRelativePose = ProxyRelativePosesGen<StdArray1>;

// example proxy: return relative poses, compute offset of rel pose from linearization points
template <template <class> class Container>
struct Proxy2DRelativePosesGen : ProxyBase {
  using InputVariablesType = std::tuple<Variable<Sophus::SE2d>*, Variable<Sophus::SE2d>*>;
  static constexpr int ResultDim = 3;
  using ResultType = Eigen::Vector<double, ResultDim>;
  using JacobianType = std::tuple<Eigen::Matrix<double, 3, 3>, Eigen::Matrix<double, 3, 3>>;

  Proxy2DRelativePosesGen(
      Variable<Sophus::SE2d>* T_p0_w,
      const Container<Variable<Sophus::SE2d>*>& T_pi_w,
      const Container<Sophus::SE2d>& linPt_T_pi_p0)
      : T_p0_w(T_p0_w), T_pi_w(T_pi_w), linPt_T_pi_p0(linPt_T_pi_p0) {
    BASPACHO_CHECK_EQ(T_pi_w.size(), linPt_T_pi_p0.size());
  }

  int64_t numEntries() const {
    return T_pi_w.size();
  }

  InputVariablesType inputVariables(int64_t i) const {
    return {T_p0_w, T_pi_w[i]};
  }

  ResultType eval(int64_t i) const {
    return (T_pi_w[i]->value * T_p0_w->value.inverse() * linPt_T_pi_p0[i].inverse()).log();
  }

  static Eigen::Matrix<double, 3, 3> SE2_leftJacobianInverse(const Eigen::Vector<double, 3>& logX) {
    Eigen::Vector<double, 6> logX6;
    logX6 << Eigen::Vector<double, 1>::Zero(), logX, Eigen::Vector<double, 2>::Zero();
    return Sophus::SE3d::leftJacobianInverse(logX6).block<3, 3>(1, 1);
  }

  ResultType eval(int64_t i, JacobianType* jac) const {
    Sophus::SE2d T_pi_p0 = T_pi_w[i]->value * T_p0_w->value.inverse();
    ResultType logDelta = (T_pi_p0 * linPt_T_pi_p0[i].inverse()).log();

    const Eigen::Matrix<double, 3, 3> dLogDelta_dLeftPi = SE2_leftJacobianInverse(logDelta);
    std::get<0>(*jac) = -dLogDelta_dLeftPi * T_pi_p0.Adj();
    std::get<1>(*jac) = dLogDelta_dLeftPi;
    return logDelta;
  }

  Variable<Sophus::SE2d>* T_p0_w;
  Container<Variable<Sophus::SE2d>*> T_pi_w;
  Container<Sophus::SE2d> linPt_T_pi_p0;
};

using Proxy2DRelativePoses = Proxy2DRelativePosesGen<StdVector>;
using Proxy2DRelativePose = Proxy2DRelativePosesGen<StdArray1>;

// return relative velocities, compute offset from velocity linearization points
template <template <class> class Container>
struct ProxyTransformedVelocitiesGen : ProxyBase {
  using InputVariablesType = std::tuple<Variable<Sophus::SE3d>*, Variable<Eigen::Vector3d>*>;
  static constexpr int ResultDim = 3;
  using ResultType = Eigen::Vector<double, ResultDim>;
  using JacobianType = std::tuple<Eigen::Matrix<double, 3, 6>, Eigen::Matrix<double, 3, 3>>;

  ProxyTransformedVelocitiesGen(
      Variable<Sophus::SE3d>* T_p0_w,
      const Container<Variable<Eigen::Vector3d>*>& iVel_w,
      const Container<Eigen::Vector3d>& linPt_iVel)
      : T_p0_w(T_p0_w), iVel_w(iVel_w), linPt_iVel(linPt_iVel) {
    BASPACHO_CHECK_EQ(iVel_w.size(), linPt_iVel.size());
  }

  int64_t numEntries() const {
    return iVel_w.size();
  }

  InputVariablesType inputVariables(int64_t i) const {
    return {T_p0_w, iVel_w[i]};
  }

  ResultType eval(int64_t i) const {
    return T_p0_w->value.so3() * iVel_w[i]->value - linPt_iVel[i];
  }

  ResultType eval(int64_t i, JacobianType* jac) const {
    Eigen::Vector3d transformed_vel = T_p0_w->value.so3() * iVel_w[i]->value;
    std::get<0>(*jac) << Eigen::Matrix3d::Zero(), Sophus::SO3d::hat(-transformed_vel);
    std::get<1>(*jac) = T_p0_w->value.so3().Adj();
    return transformed_vel - linPt_iVel[i];
  }

  Variable<Sophus::SE3d>* T_p0_w;
  Container<Variable<Eigen::Vector3d>*> iVel_w;
  Container<Eigen::Vector3d> linPt_iVel;
};

using ProxyTransformedVelocities = ProxyTransformedVelocitiesGen<StdVector>;
using ProxyTransformedVelocity = ProxyTransformedVelocitiesGen<StdArray1>;

// return relative position, compute offset from position linearization points
template <template <class> class Container>
struct ProxyTransformedPointsGen : ProxyBase {
  using InputVariablesType = std::tuple<Variable<Sophus::SE3d>*, Variable<Eigen::Vector3d>*>;
  static constexpr int ResultDim = 3;
  using ResultType = Eigen::Vector<double, ResultDim>;
  using JacobianType = std::tuple<Eigen::Matrix<double, 3, 6>, Eigen::Matrix<double, 3, 3>>;

  ProxyTransformedPointsGen(
      Variable<Sophus::SE3d>* T_loc_w,
      const Container<Variable<Eigen::Vector3d>*>& iPt_w,
      const Container<Eigen::Vector3d>& linPt_iPt)
      : T_loc_w(T_loc_w), iPt_w(iPt_w), linPt_iPt(linPt_iPt) {
    BASPACHO_CHECK_EQ(iPt_w.size(), linPt_iPt.size());
  }

  int64_t numEntries() const {
    return iPt_w.size();
  }

  InputVariablesType inputVariables(int64_t i) const {
    return {T_loc_w, iPt_w[i]};
  }

  ResultType eval(int64_t i) const {
    return T_loc_w->value * iPt_w[i]->value - linPt_iPt[i];
  }

  ResultType eval(int64_t i, JacobianType* jac) const {
    Eigen::Vector3d transformed_pt = T_loc_w->value * iPt_w[i]->value;
    std::get<0>(*jac) << Eigen::Matrix3d::Identity(), Sophus::SO3d::hat(-transformed_pt);
    std::get<1>(*jac) = T_loc_w->value.so3().matrix();
    return transformed_pt - linPt_iPt[i];
  }

  Variable<Sophus::SE3d>* T_loc_w;
  Container<Variable<Eigen::Vector3d>*> iPt_w;
  Container<Eigen::Vector3d> linPt_iPt;
};

using ProxyTransformedPoints = ProxyTransformedPointsGen<StdVector>;
using ProxyTransformedPoint = ProxyTransformedPointsGen<StdArray1>;

} // namespace small_thing
