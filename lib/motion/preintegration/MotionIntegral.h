/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sophus/se3.hpp>
#include <Eigen/Geometry>

namespace visual_inertial_ba::preintegration {

using Vec1 = Eigen::Vector<double, 1>;
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Vec6 = Eigen::Vector<double, 6>;
using Vec9 = Eigen::Vector<double, 9>;
using Mat33 = Eigen::Matrix3d;
using Mat96 = Eigen::Matrix<double, 9, 6>;
using Mat9X = Eigen::Matrix<double, 9, Eigen::Dynamic>;
template <typename T>
using Ref = Eigen::Ref<T>;
using SO3 = Sophus::SO3<double>;
using SE3 = Sophus::SE3d;

struct RotVelPos {
  SO3 R; // R_prev_next
  Vec3 dV; // accelIntegral_prev
  Vec3 dP; // accelDoubleIntegral_prev
  double dtSec;
};

/* obtain a delta vector from two RotVelPos params */
Vec9 boxMinus(const RotVelPos& a, const RotVelPos& b);

/* apply a delta vector to a RotVelPos param */
RotVelPos boxPlus(const RotVelPos& b, const Vec9& delta);

/* combine two RotVelPos over two subsequent time windows into one. */
RotVelPos combine(const RotVelPos& a, const RotVelPos& b);

/* Return `b` such that `c = combine(a, b)` */
RotVelPos uncombineLeft(const RotVelPos& c, const RotVelPos& a);

/* Return `a` such that `c = combine(a, b)` */
RotVelPos uncombineRight(const RotVelPos& c, const RotVelPos& b);

/* combine, and combined jacobians. There is not assumption on the derived params,
 * only on the image of the jacobians being RotVelPos equipped with boxPlus/minus */
RotVelPos combineJacs(
    const RotVelPos& a,
    const RotVelPos& b,
    Ref<const Mat9X> aJac,
    Ref<const Mat9X> bJac,
    Ref<Mat9X> cJac);

/* integrate gyro/accel signal over a time window. Equation is *exact* assuming the signal is
 * constant over the time window, there is no assumption that the window is small */
RotVelPos integrate(const Vec3& gyroRadSec, const Vec3& accelMSec2, double dtSec);

// As above, computing Jacobian
RotVelPos integrate(const Vec3& gyroRadSec, const Vec3& accelMSec2, double dtSec, Mat96* paramJac);

// data needed to interpolate between to RVP
struct RVPInterpolationData {
  Vec3 gyroRadSec;
  Vec3 accelMSec2;
  Vec3 deltaVelMSec; // needed to fix position mismatch
};

/* gyro/accel that would give rise to given rotation/velocity, over the given dt */
RVPInterpolationData differentiate(const RotVelPos& rvp);

// integrate interpolation data
RotVelPos integrate(const RVPInterpolationData& interp, double dtSec);

} // namespace visual_inertial_ba::preintegration
