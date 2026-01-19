/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <preintegration/PreIntegration.h>

#include <preintegration/CompensateJac.h>

#include <iostream>

namespace visual_inertial_ba::preintegration {

static int64_t measIndex_GT(const std::vector<ImuMeasurement>& meas, int64_t timeNs) {
  const auto it = std::upper_bound( // first meas >(strictly) end time
      meas.begin(),
      meas.end(),
      timeNs,
      [](int64_t tNs, const auto& m) { return tNs < timestampNs(m); });
  if (it == meas.end()) {
    throw std::runtime_error("measIndex_GT: unexpected, it == meas.end()");
  }
  return std::distance(meas.begin(), it);
}

void enumIntegrationSteps(
    const std::vector<ImuMeasurement>& meas,
    const ImuMeasurementModelParameters& measModel,
    int64_t timeStartUs,
    int64_t timeEndUs,
    const std::function<void(const SignalStatistics&, const SignalStatistics&, double, bool, bool)>&
        enumFunc) {
  const int64_t dtReferenceGyroNs = (int64_t)(measModel.dtReferenceGyroSec * 1e9);
  const int64_t dtReferenceAccelNs = (int64_t)(measModel.dtReferenceAccelSec * 1e9);
  const int64_t refTimeStartNs = timeStartUs * 1000;
  const int64_t refTimeEndNs = timeEndUs * 1000;
  const int64_t gyroTimeStartNs = refTimeStartNs + dtReferenceGyroNs;
  const int64_t gyroTimeEndNs = refTimeEndNs + dtReferenceGyroNs;
  const int64_t accelTimeStartNs = refTimeStartNs + dtReferenceAccelNs;
  const int64_t accelTimeEndNs = refTimeEndNs + dtReferenceAccelNs;

  // Make sure start/endpoint don't fall exactly on a measurement boundary, we select
  // first measurement to be > start + margin (and is left-extended by margin), and
  // last measurement to be > end - margin (and is right-extended by margin).
  // This removes ambiguity in the (extremely rare) case of start/end being exactly on a boundary
  const int64_t kMarginNs = 1'000;
  const int64_t gyroStartIndex = measIndex_GT(meas, gyroTimeStartNs + kMarginNs);
  const int64_t gyroEndIndex = measIndex_GT(meas, gyroTimeEndNs - kMarginNs);
  if (gyroStartIndex <= 0) {
    throw "enumIntegrationSteps: gyro index, not enough margin at beginning of interval";
  }

  const int64_t accelStartIndex = measIndex_GT(meas, accelTimeStartNs + kMarginNs);
  const int64_t accelEndIndex = measIndex_GT(meas, accelTimeEndNs - kMarginNs);
  if (accelStartIndex <= 0) {
    throw "enumIntegrationSteps: accel index, not enough margin at beginning of interval";
  }

  int64_t prevTimeStamp = refTimeStartNs;
  for (int64_t gi = gyroStartIndex, ai = accelStartIndex; //
       gi <= gyroEndIndex && ai <= accelEndIndex;
       /* NOOP */) {
    const auto& atGyroMeas = meas[gi];
    const auto& atAccelMeas = meas[ai];
    const auto& atPrevGyroMeas = meas[gi - 1];
    const auto& atPrevAccelMeas = meas[ai - 1];
    const int64_t adjGyroMeasStampNs = timestampNs(atGyroMeas) - dtReferenceGyroNs;
    const int64_t adjAccelMeasStampNs = timestampNs(atAccelMeas) - dtReferenceAccelNs;
    const int64_t endMeasStampNs = std::min(adjGyroMeasStampNs, adjAccelMeasStampNs);

    // check if we are transitioning into a new accel measurement
    const bool transitionToNewAccelMeas = (gi > gyroStartIndex || ai > accelStartIndex) &&
        (timestampNs(atPrevAccelMeas) - dtReferenceAccelNs == prevTimeStamp);
    const bool transitionToNewGyroMeas = (gi > gyroStartIndex || ai > accelStartIndex) &&
        (timestampNs(atPrevGyroMeas) - dtReferenceGyroNs == prevTimeStamp);

    // final timestamps is always extended to refTimeEndNs
    const int64_t endTimeStamp =
        (gi >= gyroEndIndex && ai >= accelEndIndex) ? refTimeEndNs : endMeasStampNs;
    const double dtSec = (endTimeStamp - prevTimeStamp) * 1e-9; // overlapping length
    prevTimeStamp = endTimeStamp;

    // bump indices that are ending at the interval end
    gi += (adjGyroMeasStampNs == endMeasStampNs);
    ai += (adjAccelMeasStampNs == endMeasStampNs);

    // gyro/accel signals
    const double dtGyroToPrevSec = (timestampNs(atGyroMeas) - timestampNs(atPrevGyroMeas)) * 1e-9;
    SignalStatistics uncompensatedGyroRadSec{
        .averageSignal = atGyroMeas.gyroRadSec.cast<double>(),
        .rate =
            (atGyroMeas.gyroRadSec - atPrevGyroMeas.gyroRadSec).cast<double>() / dtGyroToPrevSec,
    };
    const double dtAccelToPrevSec =
        (timestampNs(atAccelMeas) - timestampNs(atPrevAccelMeas)) * 1e-9;
    SignalStatistics uncompensatedAccelMSec2{
        .averageSignal = atAccelMeas.accelMSec2.cast<double>(),
        .rate =
            (atAccelMeas.accelMSec2 - atPrevAccelMeas.accelMSec2).cast<double>() / dtAccelToPrevSec,
    };

    enumFunc(
        uncompensatedGyroRadSec,
        uncompensatedAccelMSec2,
        dtSec,
        transitionToNewAccelMeas,
        transitionToNewGyroMeas);
  }
}

// derivative of `rvp` wrt integration interval extreme changes
static Vec9
dRvp_dLeftCompensatedMeas(const RotVelPos& rvp, const Vec3& gyroRadSec, const Vec3& accelMSec2) {
  Vec9 dRvp;
  dRvp << gyroRadSec, //
      SO3::hat(-rvp.dV) * gyroRadSec + accelMSec2, //
      accelMSec2 * rvp.dtSec + SO3::hat(-rvp.dP) * gyroRadSec;
  return dRvp;
}

static Vec9
dRvp_dStartTime(const RotVelPos& rvp, const Vec3& startGyroRadSec, const Vec3& startAccelMSec2) {
  return dRvp_dLeftCompensatedMeas(rvp, -startGyroRadSec, -startAccelMSec2);
}

static Vec9
dRvp_dEndTime(const RotVelPos& rvp, const Vec3& endGyroRadSec, const Vec3& endAccelMSec2) {
  const Mat33 Rmat = rvp.R.matrix();
  Vec9 dRvp;
  dRvp << Rmat * endGyroRadSec, Rmat * endAccelMSec2, rvp.dV;
  return dRvp;
}

PreIntegration computePreIntegration(
    const ImuCalibrationJacobianIndices& jacInd,
    const std::vector<ImuMeasurement>& meas,
    const ImuMeasurementModelParameters& measModel,
    const ImuNoiseModelParameters& noiseModel,
    int64_t timeStartUs,
    int64_t timeEndUs) {
  const size_t eStateSize = jacInd.getErrorStateSize();

  std::vector<RotVelPos> pertRvp(eStateSize);
  std::optional<RotVelPos> maybeRvp = {};

  /* We compute
   *   rvp = rvp1 (+) rvp2, and
   *   rvpJac as function of rvp1Jac, rvp2Jac,
   * where rvp2 was compute from compMeas2, in turn from (rawMeas2, calib)
   * we select
   *   rvp1Jac = (1 | 0              | rvp1CalibJac)
   *   rvp2Jac = (0 | rvp2RawMeasJac | rvp2CalibJac)
   * so that we can use rvp2 to get covariance from rvp1noise and rvp2noise,
   * assumed to be independent.
   */
  Mat9X rvpJac(9, 15 + eStateSize), rvp1Jac(9, 15 + eStateSize), rvp2Jac(9, 15 + eStateSize);

  using Mat93 = Eigen::Matrix<double, 9, 3>;
  Mat99 rvpCov = Mat99::Zero();
  Mat93 rvpFromAccelNoise = Mat93::Zero();
  Mat93 rvpFromGyroNoise = Mat93::Zero();

  Mat6X compMeasJac(6, 6 + eStateSize); // Jac of: compMeas(rawMeas, calib)
  Vec3 startGyroRadSec, startAccelMSec2; // will save first compensated meas
  Vec3 prevAccelMSec2, prevGyroRadSec; // needed to compute gyro-accel time-offset jac
  SignalStatistics prevRawAccelMSec2, prevRawGyroRadSec; // for rare case of aligned accel/gyro
  Vec3 gyroRadSec, accelMSec2; // will hold last value
  double totDT = 0;

  enumIntegrationSteps(
      meas,
      measModel,
      timeStartUs,
      timeEndUs,
      [&](const SignalStatistics& rawGyroRadSec,
          const SignalStatistics& rawAccelMSec2,
          double dtSec,
          bool transitionToNewAccelMeas,
          bool transitionToNewGyroMeas) {
        totDT += dtSec;
        getCompensatedImuMeasurementAndJac(
            measModel,
            rawGyroRadSec,
            rawAccelMSec2,
            gyroRadSec,
            accelMSec2,
            jacInd,
            compMeasJac.rightCols(eStateSize), // calib
            compMeasJac.leftCols<6>()); // rawMeas

        Mat96 rvpCompMeasJac; // rvp(compMeas2)
        auto rvp = integrate(gyroRadSec, accelMSec2, dtSec, &rvpCompMeasJac);
        rvp2Jac.leftCols<9>().setZero(); // rvp2 does not depend on rvp1
        rvp2Jac.rightCols(6 + eStateSize) = rvpCompMeasJac * compMeasJac; // rvp(rawMeas2, calib)

        if (transitionToNewAccelMeas && jacInd.gyroAccelTimeOffsetIdx()) {
          // If transitioning to new accel measurement, the variation given by sliding the boundary
          // to the left is like applying (on the left) a (newMeas - prevMeas) x dt
          Vec3 deltaGyro = gyroRadSec - prevGyroRadSec;
          Vec3 deltaAccel = accelMSec2 - prevAccelMSec2;
          if (transitionToNewGyroMeas) {
            Vec3 backGyroRadSec, backAccelMSec2; // sliding accel backward
            Vec3 forwGyroRadSec, forwAccelMSec2; // sliding accel forward
            measModel.getCompensatedImuMeasurement(
                rawGyroRadSec, prevRawAccelMSec2, forwGyroRadSec, forwAccelMSec2);
            measModel.getCompensatedImuMeasurement(
                prevRawGyroRadSec, rawAccelMSec2, backGyroRadSec, backAccelMSec2);
            deltaGyro = (backGyroRadSec - prevGyroRadSec + gyroRadSec - forwGyroRadSec) * 0.5;
            deltaAccel = (backAccelMSec2 - prevAccelMSec2 + accelMSec2 - forwAccelMSec2) * 0.5;
          }
          rvp2Jac.col(15 + jacInd.gyroAccelTimeOffsetIdx()) =
              dRvp_dLeftCompensatedMeas(rvp, deltaGyro, deltaAccel);
        }
        prevAccelMSec2 = accelMSec2;
        prevGyroRadSec = gyroRadSec;
        prevRawAccelMSec2 = rawAccelMSec2;
        prevRawGyroRadSec = rawGyroRadSec;

        if (!maybeRvp.has_value()) {
          maybeRvp = rvp;
          std::swap(rvpJac, rvp2Jac);

          // save starting gyro/accel to compute ref time offset's jacobian
          startGyroRadSec = gyroRadSec;
          startAccelMSec2 = accelMSec2;
        } else {
          std::swap(rvpJac, rvp1Jac);
          rvp1Jac.leftCols<9>().setIdentity(); // Jac of: rvp1(rvp1)
          rvp1Jac.middleCols<6>(9).setZero(); // (rvp1 does not depend on rawMeas2)
          maybeRvp = combineJacs(maybeRvp.value(), rvp, rvp1Jac, rvp2Jac, rvpJac);
        }

        // J's first 6 cols are now a map from added raw measurements to rvp
        auto rvpRvp1Jac = rvpJac.leftCols<9>();
        rvpCov = rvpRvp1Jac * rvpCov * rvpRvp1Jac.transpose();
        rvpFromGyroNoise = rvpRvp1Jac * rvpFromGyroNoise;
        rvpFromAccelNoise = rvpRvp1Jac * rvpFromAccelNoise;
        if (transitionToNewGyroMeas) { // prev gyro noise is independent of current noise
          rvpCov += rvpFromGyroNoise * noiseModel.gyroSampleVarianceRad2Sec2.asDiagonal() *
              rvpFromGyroNoise.transpose();
          rvpFromGyroNoise.setZero();
        }
        if (transitionToNewAccelMeas) { // prev accel noise is independent of current noise
          rvpCov += rvpFromAccelNoise * noiseModel.accelSampleVarianceM2Sec4.asDiagonal() *
              rvpFromAccelNoise.transpose();
          rvpFromAccelNoise.setZero();
        }

        // noise components from jac of rvp(rawMeas2)
        rvpFromGyroNoise += rvpJac.middleCols<3>(9);
        rvpFromAccelNoise += rvpJac.middleCols<3>(12);
      });
  rvpCov += rvpFromGyroNoise * noiseModel.gyroSampleVarianceRad2Sec2.asDiagonal() *
      rvpFromGyroNoise.transpose();
  rvpCov += rvpFromAccelNoise * noiseModel.accelSampleVarianceM2Sec4.asDiagonal() *
      rvpFromAccelNoise.transpose();

  // ref imu time offset's jac is computed here
  MatX J = rvpJac.rightCols(eStateSize);
  if (jacInd.referenceImuTimeOffsetIdx() >= 0) {
    J.col(jacInd.referenceImuTimeOffsetIdx()) =
        dRvp_dStartTime(*maybeRvp, startGyroRadSec, startAccelMSec2) +
        dRvp_dEndTime(*maybeRvp, gyroRadSec, accelMSec2);
  }

  return {
      .rvp = maybeRvp.value(),
      .J = J,
      .rvpCov = rvpCov,
      .omegaAtEnd = prevGyroRadSec,
      .calibEvalPoint = measModel,
  };
}

RotVelPos integrateMeasurements(
    const std::vector<ImuMeasurement>& meas,
    const ImuMeasurementModelParameters& measModel,
    int64_t timeStartUs,
    int64_t timeEndUs) {
  std::optional<RotVelPos> maybeRvp = {};

  enumIntegrationSteps(
      meas,
      measModel,
      timeStartUs,
      timeEndUs,
      [&](const SignalStatistics& uncompensatedGyroRadSec,
          const SignalStatistics& uncompensatedAccelMSec2,
          double dtSec,
          bool /* transitionToNewAccelMeas */,
          bool /* transitionToNewGyroMeas */
      ) {
        Vec3 gyroRadSec, accelMSec2;
        measModel.getCompensatedImuMeasurement(
            uncompensatedGyroRadSec, uncompensatedAccelMSec2, gyroRadSec, accelMSec2);
        auto rvp = integrate(gyroRadSec, accelMSec2, dtSec);
        if (!maybeRvp.has_value()) {
          maybeRvp = rvp;
        } else {
          maybeRvp = combine(maybeRvp.value(), rvp);
        }
      });

  return *maybeRvp;
}

void forEachIntegratedMeasurement(
    const std::vector<ImuMeasurement>& meas,
    const ImuMeasurementModelParameters& measModel,
    int64_t timeStartUs,
    int64_t timeEndUs,
    const std::function<void(const RotVelPos&, bool, bool)>& enumFunc) {
  RotVelPos prevRvp = {
      .R = SO3(),
      .dV = Vec3::Zero(),
      .dP = Vec3::Zero(),
      .dtSec = 0.0,
  };

  enumIntegrationSteps(
      meas,
      measModel,
      timeStartUs,
      timeEndUs,
      [&](const SignalStatistics& uncompensatedGyroRadSec,
          const SignalStatistics& uncompensatedAccelMSec2,
          double dtSec,
          bool transitionToNewAccelMeas,
          bool transitionToNewGyroMeas) {
        bool atStart = (prevRvp.dtSec == 0.0);
        enumFunc(prevRvp, transitionToNewAccelMeas || atStart, transitionToNewGyroMeas || atStart);

        Vec3 gyroRadSec, accelMSec2;
        measModel.getCompensatedImuMeasurement(
            uncompensatedGyroRadSec, uncompensatedAccelMSec2, gyroRadSec, accelMSec2);
        auto rvp = integrate(gyroRadSec, accelMSec2, dtSec);
        prevRvp = combine(prevRvp, rvp);
      });

  enumFunc(prevRvp, true, true);
}

} // namespace visual_inertial_ba::preintegration
