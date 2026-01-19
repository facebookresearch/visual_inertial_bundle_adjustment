/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sophus/se3.hpp>
#include <viba/common/FindOrDie.h>
#include <viba/common/Format.h>
#include <Eigen/Geometry>

namespace visual_inertial_ba {

using RigSensorIndices = std::pair<int64_t, int>;
using RigCamIndices = RigSensorIndices;
using RigImuIndices = RigSensorIndices;

template <typename T>
using Ref = Eigen::Ref<T>;

template <int R>
using Vec = Eigen::Vector<double, R>;
using Vec2 = Vec<2>;
using Vec3 = Vec<3>;
using Vec4 = Vec<4>;
using Vec6 = Vec<6>;
using Vec7 = Vec<7>;
using Vec9 = Vec<9>;
using VecX = Eigen::VectorXd;

template <int R, int C>
using Mat = Eigen::Matrix<double, R, C>;
using Mat22 = Mat<2, 2>;
using Mat32 = Mat<3, 2>;
using Mat33 = Mat<3, 3>;
using Mat36 = Mat<3, 6>;
using Mat63 = Mat<6, 3>;
using Mat66 = Mat<6, 6>;
using Mat23 = Mat<2, 3>;
using Mat26 = Mat<2, 6>;
using Mat29 = Mat<2, 9>;
using Mat92 = Mat<9, 2>;
using Mat93 = Mat<9, 3>;
using Mat96 = Mat<9, 6>;
using Mat99 = Mat<9, 9>;
using Mat1X = Mat<1, Eigen::Dynamic>;
using Mat3X = Mat<3, Eigen::Dynamic>;
using Mat6X = Mat<6, Eigen::Dynamic>;
using Mat9X = Mat<9, Eigen::Dynamic>;
using MatX = Eigen::MatrixXd;

using SE3 = Sophus::SE3d;
using SO3 = Sophus::SO3d;

using Vec2f = Eigen::Vector<float, 2>;
using Mat23f = Eigen::Matrix<float, 2, 3>;

template <typename T>
using Map = Eigen::Map<T>;
template <typename T>
using MapK = Eigen::Map<const T>;

constexpr inline double degreesToRadians(double x) {
  return x * (M_PI / 180.0);
}

constexpr inline double radiansToDegrees(double x) {
  return x * (180.0 / M_PI);
}

} // namespace visual_inertial_ba
