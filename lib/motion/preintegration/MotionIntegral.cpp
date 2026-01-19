/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <preintegration/MotionIntegral.h>

#include <iostream>

namespace visual_inertial_ba::preintegration {

Vec9 boxMinus(const RotVelPos& a, const RotVelPos& b) {
  Vec9 retv;
  retv << (a.R * b.R.inverse()).log(), a.dV - b.dV, a.dP - b.dP;
  return retv;
}

RotVelPos boxPlus(const RotVelPos& b, const Vec9& delta) {
  return RotVelPos{
      .R = SO3::exp(delta.template head<3>()) * b.R,
      .dV = delta.template segment<3>(3) + b.dV,
      .dP = delta.template tail<3>() + b.dP,
  };
}

RotVelPos combine(const RotVelPos& a, const RotVelPos& b) {
  const SO3 R = a.R * b.R;
  const Vec3 dV = a.dV + a.R * b.dV;
  const Vec3 dP = a.dP + a.dV * b.dtSec + a.R * b.dP;
  return {R, dV, dP, a.dtSec + b.dtSec};
}

RotVelPos uncombineLeft(const RotVelPos& c, const RotVelPos& a) {
  const SO3 a_R_inv = a.R.inverse();
  const SO3 b_R = a_R_inv * c.R;
  const Vec3 b_dV = a_R_inv * (c.dV - a.dV);
  const double b_dtSec = c.dtSec - a.dtSec;
  const Vec3 b_dP = a_R_inv * (c.dP - a.dP - a.dV * b_dtSec);
  return {b_R, b_dV, b_dP, b_dtSec};
}

RotVelPos uncombineRight(const RotVelPos& c, const RotVelPos& b) {
  const SO3 a_R = c.R * b.R.inverse();
  const Vec3 a_dV = c.dV - a_R * b.dV;
  const double a_dtSec = c.dtSec - b.dtSec;
  const Vec3 a_dP = c.dP - a_dV * b.dtSec - a_R * b.dP;
  return {a_R, a_dV, a_dP, a_dtSec};
}

RotVelPos combineJacs(
    const RotVelPos& a,
    const RotVelPos& b,
    Ref<const Mat9X> aJac,
    Ref<const Mat9X> bJac,
    Ref<Mat9X> cJac) {
  const Vec3 aRbV = a.R * b.dV;
  const Vec3 aRbP = a.R * b.dP;
  const SO3 R = a.R * b.R;
  const Vec3 dV = a.dV + aRbV;
  const Vec3 dP = a.dP + a.dV * b.dtSec + aRbP;
  const Mat33 aRmat = a.R.matrix(); // =Adj for SO3
  cJac.setZero();
  cJac.topRows<3>() = aJac.topRows<3>() //
      + aRmat * bJac.topRows<3>(); // a.R.Adj() * ...
  cJac.middleRows<3>(3) = aJac.middleRows<3>(3) //
      + SO3::hat(-aRbV) * aJac.topRows<3>() //
      + aRmat * bJac.middleRows<3>(3);
  cJac.bottomRows<3>() = aJac.bottomRows<3>() //
      + aJac.middleRows<3>(3) * b.dtSec //
      + SO3::hat(-aRbP) * aJac.topRows<3>() //
      + aRmat * bJac.bottomRows<3>();
  return {R, dV, dP, a.dtSec + b.dtSec};
}

// factorials, for convenience
constexpr double F2 = 2.0;
constexpr double F3 = 6.0;
constexpr double F4 = 24.0;
constexpr double F5 = 120.0;
constexpr double F6 = 729.0;
constexpr double F7 = 5040.0;
constexpr double F8 = 40320.0;
constexpr double F9 = 362880;
constexpr double F10 = 3628800;

RVPInterpolationData differentiate(const RotVelPos& rvp) {
  const Vec3 omega = rvp.R.log();
  const double th2 = omega.squaredNorm();
  const double th = std::sqrt(th2);

  // rot(1-q2*th2, q1*th) = rot(1-c2*th2, c1*th)^{-1}
  double q1 = -0.5, q2;
  if (th < 1e-3) {
    // tayor expansion of (1.0 - h * cos(h) / sin(h)) / (4h^2)
    constexpr double K0 = 12.0;
    constexpr double K2 = 180.0;
    constexpr double K4 = 1890.0;
    q2 = 1.0 / K0 - th2 / (4.0 * K2) + (th2 * th2) / (16.0 * K4);
  } else {
    const double h = th * 0.5;
    q2 = (1.0 - h * cos(h) / sin(h)) / th2;
  }

  const Vec3 omegaVel = omega.cross(rvp.dV);
  const Vec3 upsilon = rvp.dV + q1 * omegaVel + q2 * omega.cross(omegaVel);

  const RotVelPos reconRvp = integrate(omega / rvp.dtSec, upsilon / rvp.dtSec, rvp.dtSec);
  return RVPInterpolationData{
      .gyroRadSec = omega / rvp.dtSec, // over 1 sec
      .accelMSec2 = upsilon / rvp.dtSec, // over 1 sec
      .deltaVelMSec = (rvp.dP - reconRvp.dP) / rvp.dtSec,
  };
}

RotVelPos integrate(const RVPInterpolationData& interp, double dtSec) {
  auto rvp = integrate(interp.gyroRadSec, interp.accelMSec2, dtSec);
  rvp.dP += interp.deltaVelMSec * dtSec;
  return rvp;
}

RotVelPos integrate(const Vec3& gyroRadSec, const Vec3& accelMSec2, double dtSec) {
  const Vec3 omega = gyroRadSec * dtSec;
  const Vec3 upsilon = accelMSec2 * dtSec;

  const SO3 R = SO3::exp(omega);

  const double th2 = omega.squaredNorm();
  const double th = std::sqrt(th2);
  const double th4 = th2 * th2;

  double c1, c2, c3;
  if (th < 1e-3) {
    c1 = (1.0 / F2) - (th2 / F4) + (th4 / F6);
    c2 = (1.0 / F3) - (th2 / F5) + (th4 / F7);
    c3 = (1.0 / F4) - (th2 / F6) + (th4 / F8);
  } else {
    const double sThDTh = sin(th) / th;
    const double mCThDTh2 = (1.0 - cos(th)) / th2;
    c1 = mCThDTh2;
    c2 = (1.0 - sThDTh) / th2;
    c3 = (0.5 - mCThDTh2) / th2;
  }

  const Mat33 Omega = SO3::hat(omega);
  const Mat33 Omega_sq = Omega * Omega;

  // v = [1 + k2(th) * (Omega/th) + (1-k1(th)) * (Omega_sq / th2)] * upsilon
  const Mat33 U2V = Mat33::Identity() + c1 * Omega + c2 * Omega_sq;
  const Vec3 dV = U2V * upsilon;

  // v = [th + k2(th)*th * (Omega/th) + (1-k1(th))*th * (Omega_sq / th2)]
  //         * (accelMSec2 * (dtSec/th))
  // p = int v(Theta) (dTheta/dTime) dTime
  const Mat33 U2P = Mat33::Identity() / 2.0 + c2 * Omega + c3 * Omega_sq;
  const Vec3 dP = U2P * (upsilon * dtSec);

  return {R, dV, dP, dtSec};
}

RotVelPos integrate(const Vec3& gyroRadSec, const Vec3& accelMSec2, double dtSec, Mat96* paramJac) {
  const Vec3 omega = gyroRadSec * dtSec;
  const Vec3 upsilon = accelMSec2 * dtSec;

  const SO3 R = SO3::exp(omega);

  const double th2 = omega.squaredNorm();
  const double th = std::sqrt(th2);
  const double th4 = th2 * th2;

  double c1, c2, c3, d1, d2, d3;
  if (th < 1e-3) {
    c1 = (1.0 / F2) - (th2 / F4) + (th4 / F6);
    c2 = (1.0 / F3) - (th2 / F5) + (th4 / F7);
    c3 = (1.0 / F4) - (th2 / F6) + (th4 / F8);
    d1 = -(2.0 / F4) + th2 * (4.0 / F6) + th4 * (6.0 / F8);
    d2 = -(2.0 / F5) + th2 * (4.0 / F7) + th4 * (6.0 / F9);
    d3 = -(2.0 / F6) + th2 * (4.0 / F8) + th4 * (6.0 / F10);
  } else {
    const double sThDTh = sin(th) / th;
    const double mCThDTh2 = (1.0 - cos(th)) / th2;
    c1 = mCThDTh2;
    c2 = (1.0 - sThDTh) / th2;
    c3 = (0.5 - mCThDTh2) / th2;
    d1 = (sThDTh - 2.0 * mCThDTh2) / th2;
    d2 = (mCThDTh2 - 3.0 * c2) / th2;
    d3 = (-1.0 - sThDTh + 4.0 * mCThDTh2) / th4;
  }

  const Mat33 Omega = SO3::hat(omega);
  const Mat33 Omega_sq = Omega * Omega;

  // v = [1 + k2(th) * (Omega/th) + (1-k1(th)) * (Omega_sq / th2)] * upsilon
  const Mat33 U2V = Mat33::Identity() + c1 * Omega + c2 * Omega_sq;
  const Vec3 dV = U2V * upsilon;

  // v = [th + k2(th)*th * (Omega/th) + (1-k1(th))*th * (Omega_sq / th2)]
  //         * (accelMSec2 * (dtSec/th))
  // p = int v(Theta) (dTheta/dTime) dTime
  const Mat33 U2P = Mat33::Identity() / 2.0 + c2 * Omega + c3 * Omega_sq;
  const Vec3 dP = U2P * (upsilon * dtSec);

  // compute jacobian
  paramJac->setZero();
  paramJac->template block<3, 3>(0, 0) = U2V * dtSec;

  const Mat33 DwXu_Dw = SO3::hat(-upsilon) * dtSec;
  const Mat33 DwXwXu_Dw = SO3::hat(-omega.cross(upsilon)) * dtSec + Omega * DwXu_Dw;

  const Vec3 V_D1 = (d1 * Omega + d2 * Omega_sq) * upsilon;
  const Mat33 JV = (V_D1 * omega.transpose()) * dtSec; // * dtheta_dgyro
  const Mat33 JV2 = c1 * DwXu_Dw + c2 * DwXwXu_Dw;
  paramJac->template block<3, 3>(3, 0) = JV + JV2;

  const Vec3 P_D1 = (d2 * Omega + d3 * Omega_sq) * (upsilon * dtSec);
  const Mat33 JP = (P_D1 * omega.transpose()) * dtSec; // * dtheta_dgyro
  const Mat33 JP2 = (c2 * DwXu_Dw + c3 * DwXwXu_Dw) * dtSec;
  paramJac->template block<3, 3>(6, 0) = JP + JP2;

  paramJac->template block<3, 3>(0, 3).setZero();
  paramJac->template block<3, 3>(3, 3) = U2V * dtSec;
  paramJac->template block<3, 3>(6, 3) = U2P * dtSec * dtSec;

  return {R, dV, dP, dtSec};
}

} // namespace visual_inertial_ba::preintegration
