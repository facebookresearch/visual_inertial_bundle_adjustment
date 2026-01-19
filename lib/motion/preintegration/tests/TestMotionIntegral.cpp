/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <preintegration/MotionIntegral.h>
#include <iostream>
#include <random>

using namespace visual_inertial_ba::preintegration;

template <typename T>
static T randVec(std::mt19937& g) {
  T retv;
  for (int i = 0; i < retv.size(); i++) {
    retv.data()[i] = std::normal_distribution<>(0, 1)(g);
  }
  return retv;
}

template <typename T>
static T randVecCap(std::mt19937& g, double cap) {
  T retv;
  for (int i = 0; i < retv.size(); i++) {
    retv.data()[i] = std::normal_distribution<>(0, 1)(g);
  }
  if (retv.norm() > cap) {
    retv *= cap / retv.norm();
  }
  return retv;
}

// test boxPlus / boxMinus being inverse of one another
TEST(TestMotionIntegral, BoxOps) {
  std::mt19937 gen(42);
  for (int i = 0; i < 100; i++) {
    Vec3 gyro = randVec<Vec3>(gen);
    Vec3 accel = randVec<Vec3>(gen);
    auto c = integrate(gyro, accel, 1.);

    // test box minus / box plus
    Vec9 delta = randVec<Vec9>(gen);
    if (delta.head<3>().norm() > M_PI * 3 / 4) { // don't wrap around
      delta.head<3>() = delta.head<3>().normalized() * M_PI * 3 / 4;
    }
    Vec9 cmpDl = boxMinus(boxPlus(c, delta), c);
    EXPECT_NEAR((cmpDl - delta).norm(), 0.0, 1e-10);
  }
}

TEST(TestMotionIntegral, Combine) {
  std::mt19937 gen(42);
  for (int i = 0; i < 100; i++) {
    Vec3 gyro = randVec<Vec3>(gen);
    Vec3 accel = randVec<Vec3>(gen);
    double t1 = std::uniform_real_distribution<>(0.1, 0.9)(gen);
    auto a = integrate(gyro, accel, t1);
    auto b = integrate(gyro, accel, 1. - t1);
    auto d = combine(a, b);
    auto c = integrate(gyro, accel, 1.);
    Vec9 err = boxMinus(d, c);
    EXPECT_NEAR(err.norm(), 0.0, 1e-10);
  }
}

TEST(TestMotionIntegral, Differentiate) {
  std::mt19937 gen(42);
  for (int i = 0; i < 100; i++) {
    Vec3 gyro = randVecCap<Vec3>(gen, 0.5);
    Vec3 accel = randVec<Vec3>(gen);
    if (i & 1) {
      gyro *= 1e-4 / gyro.norm();
    }
    if (i & 2) {
      accel *= 1e-4 / accel.norm();
    }
    auto rvp = integrate(gyro, accel, 1.0);
    Vec3 dPos = randVecCap<Vec3>(gen, 0.1);
    rvp.dP += dPos;
    auto interpData = differentiate(rvp);
    EXPECT_NEAR((gyro - interpData.gyroRadSec).norm(), 0.0, 1e-10);
    EXPECT_NEAR((accel - interpData.accelMSec2).norm(), 0.0, 1e-10);
    EXPECT_NEAR((dPos - interpData.deltaVelMSec).norm(), 0.0, 1e-10);
  }
}

TEST(TestMotionIntegral, Uncombine) {
  std::mt19937 gen(42);
  for (int i = 0; i < 100; i++) {
    Vec3 gyro1 = randVecCap<Vec3>(gen, 0.5);
    Vec3 accel1 = randVec<Vec3>(gen);
    Vec3 gyro2 = randVecCap<Vec3>(gen, 0.5);
    Vec3 accel2 = randVec<Vec3>(gen);
    double t1 = std::uniform_real_distribution<>(0.1, 0.9)(gen);
    double t2 = std::uniform_real_distribution<>(0.1, 0.9)(gen);
    auto a = integrate(gyro1, accel1, t1);
    auto b = integrate(gyro2, accel2, t2);
    auto c = combine(a, b);
    auto revB = uncombineLeft(c, a);
    auto revA = uncombineRight(c, b);
    EXPECT_NEAR(revA.dtSec, t1, 1e-10);
    EXPECT_NEAR(revB.dtSec, t2, 1e-10);
    EXPECT_NEAR(boxMinus(a, revA).norm(), 0.0, 1e-8);
    EXPECT_NEAR(boxMinus(b, revB).norm(), 0.0, 1e-8);
  }
}

TEST(TestMotionIntegral, SmallSteps) {
  std::mt19937 gen(42);
  for (int i = 0; i < 100; i++) {
    Vec3 gyro = randVec<Vec3>(gen);
    Vec3 accel = randVec<Vec3>(gen);
    const int STEPS = 10000;
    auto step = integrate(gyro, accel, 1.0 / STEPS);
    auto acc = step;
    for (int i = 1; i < STEPS; i++) {
      acc = combine(acc, step);
    }
    auto c = integrate(gyro, accel, 1.);
    Vec9 err2 = boxMinus(acc, c);
    EXPECT_NEAR(err2.norm(), 0.0, 1e-10);
  }
}

// numeric jacobian
Mat96 integrateNumJac(const Vec3& gyroRadSec, const Vec3& accelMSec2, double dtSec) {
  const double EPS = 1e-7;
  auto a = integrate(gyroRadSec, accelMSec2, dtSec);
  Mat96 retv;
  for (int i = 0; i < 6; i++) {
    Vec3 gyroRadSecP = gyroRadSec;
    Vec3 accelMSec2P = accelMSec2;
    if (i < 3) {
      gyroRadSecP[i] += EPS;
    } else {
      accelMSec2P[i - 3] += EPS;
    }
    auto b = integrate(gyroRadSecP, accelMSec2P, dtSec);
    retv.col(i) = (boxMinus(b, a) / EPS).template head<9>();
  }
  return retv;
}

TEST(TestMotionIntegral, Jacobian) {
  std::mt19937 gen(42);
  for (int i = 0; i < 100; i++) {
    Vec3 gyro = randVec<Vec3>(gen);
    Vec3 accel = randVec<Vec3>(gen);
    double t1 = std::uniform_real_distribution<>(0.1, 0.9)(gen);
    auto numJ = integrateNumJac(gyro, accel, t1);
    Mat96 anJ;
    integrate(gyro, accel, t1, &anJ);
    const Mat96 deltaJ = numJ - anJ;
    EXPECT_NEAR(deltaJ.norm(), 0.0, 1e-7);
  }
}

TEST(TestMotionIntegral, CombineJacobian) {
  std::mt19937 gen(42);
  for (int i = 0; i < 100; i++) {
    Vec3 gyro = randVec<Vec3>(gen);
    Vec3 accel = randVec<Vec3>(gen);
    double t1 = std::uniform_real_distribution<>(0.1, 0.9)(gen);

    Mat96 aJ, bJ, cJ, dJ;
    auto a = integrate(gyro, accel, t1, &aJ);
    auto b = integrate(gyro, accel, 1. - t1, &bJ);
    integrate(gyro, accel, 1., &cJ);
    combineJacs(a, b, aJ, bJ, dJ);
    EXPECT_NEAR((cJ - dJ).norm(), 0.0, 1e-10);
  }
}
