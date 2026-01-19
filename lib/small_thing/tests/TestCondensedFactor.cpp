/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-VectorFirstElementAddressOf
#include <small_thing/CondensedFactor.h>

#include <baspacho/testing/TestingUtils.h>
#include <gtest/gtest.h>
#include <small_thing/Optimizer.h>
#include <small_thing/Proxies.h>
#include <chrono>
#include <iomanip>
#include <random>

using namespace small_thing;
using namespace BaSpaCho::testing_utils;
using namespace std;

/*
  This test is also maent as a demo of how SmallThing allows us to use complex "marginalized"
  factors to simplify an optimization problem and carry on a "condensed" factor (ie a factor with
  H,b representing a quadratic model).
  What is going on is the following:
  * in Problem1, a set of "surviving" variables {s_i} is selected, which will "survive" the
    marginalization process where the remaining variables {e_i} are marginalized. The "marginal
    problem" (H, b, and cost base) are extracted, at some linearization point. The marginal problem
    is a quadratic model of the cost AS FUNCTIONG OF THE {s_i}, ASSUMING THE {e_i} ARE OPTIMIZED
    ACCORDINGLY ie keeping the {s_i} fixed.
      This means that the cost on a potential perturbation of the {s_i} is represented by a formula
                         x.T * H * x / 2  +  x.T * b  +  costBase
    where x is the vector obtained concatenating all perturbations applied to {s_i}.
      In order to evaluate the above expression we need to be able to compute x from a perturbed
    values, therefore we need the linearization points. x is obtained via "boxMinus", which extracts
    the tangent vector at the linearization point, so that when applied with "boxPlus" gives us the
    pertubed varibles.
  * In Problem2 we want to use the above expression with (H,b,costBase) as a cost term to connect
    some variables. This is made a bit more complex because the vector "x" may not be compute
    directly from the variables in Problem2, as there may not be a variable corresponding exactly to
    a surviving variable in Problem1.
      A concrete example is the following: in Problem1 we want to get the marginal problem for a
    dense block connecting with N poses, but to be more economic we set (pose1) to be constant with
    identity value, and (pose2, ..., poseN) to be variables. In this way we get a 6x6 H matrix for
    an edge, for instance.
      Now in Problem2 we have N poses, but pose1 is not set fixed to be the origin. Therefore we get
    a cost term which can be used with x representing the perturbation of the RELATIVE poses (poseJ
    to pose1, J=2,..,N). Mathematically it's easy because the (N-1) relative poses can be
    differentiably computed from the N absolute poses, we can compute they jacobian J, and indeed
    internally the optimizer will replace H, b with (J.T * H * J, J.T * b) in order to have the
    Hessian/gradient blocks corresponding to the variables present in Problem2.
      How do we make things easy on the engineering side? SmallThing offers "Proxies" (essentially
    functions with jacobian, parametrized) which can be applied to a dynamic amount of variables to
    compute the intermediate variables mapping to Problem1 (ie the relative poses from the absolute
    poses,in the example above).
      In SmallThing the way we add a dense block to N poses with an H,b marginal problem
    representing the quadratic cost on the relative poses is the following:
        optimizer.addCondensedFactor(
          H,
          b,
          costBase,
          ProxyRelativePoses(
            &var_T_rig1_world // reference pose, optimizer variable
            { &var_T_rig2_world, ..., &var_T_rigN_world }, // N - 1 optimizer variables
            { linPt_T_rig2_pose1, ... linPt_T_rigN_rig1 } // linearization points
            )
          );
      Note that multiple proxies can be used, eg if the marginal variables contain velocities we
    might want to transform N poses to N-1 relative poses, and N velocities to N transformed
    velocities in pose0's frame of reference. In this case H's order is expected to be
      6*(N-1) + 3*N
    being 6, 3 the dimensions of parametrization of SE3 and R3.
*/

using Vec1 = Eigen::Vector<double, 1>;
using Mat11 = Eigen::Matrix<double, 1, 1>;
using Mat66 = Eigen::Matrix<double, 6, 6>;
template <typename T>
using Ref = Eigen::Ref<T>;

TEST(TestCondensedFactor, Simple) {
  // points connected with springs
  static constexpr double springLen = 1.0;
  vector<Variable<Vec1>> vars = {{-2}, {-1}, {0}, {0.5}, {1.5}, {2.5}};

  // build problem with springs
  Optimizer opt;
  for (size_t i = 0; i < vars.size() - 1; i++) {
    // springs with length = `springLen`
    opt.addFactor(
        [=](const Vec1& x, const Vec1& y, Ref<Mat11>&& dx, Ref<Mat11>&& dy) -> Vec1 {
          if (!isNull(dx)) {
            dx(0, 0) = -1;
          }
          if (!isNull(dy)) {
            dy(0, 0) = 1;
          }
          return Vec1(y[0] - x[0] - springLen);
        },
        vars[i],
        vars[i + 1]);
  }

  // vars[2] will be set to constant, compute cost for reference
  vars[2].setConstant(true);
  double baseCost = opt.computeCost();
  std::cout << "Base cost: " << baseCost << std::endl;

  // compute marginal on v0, v4, v5, assuming v2 is fixed
  opt.registerAllVariables();
  auto mProb1 =
      opt.computeMarginalProblem({.damping = 0}, {vars[0].index, vars[4].index, vars[5].index});
  std::cout << "[Marg1] H: " << mProb1.H.rows() << "x" << mProb1.H.cols()
            << "; b: " << mProb1.b.size() << "; c: " << mProb1.cost << std::endl;

  // "Marginally-optimal cost", ie cost computed fixing marginal (surviving) variables and
  // optimizing the eliminated variables to their "best". Since v2=0 and v4=1.5 are fixed,
  // this is obtained setting v3=0.75
  vars[3].value[0] = 0.75;
  double marginallyOptimalCost = opt.computeCost();
  std::cout << "'marginally optimal' cost:" << marginallyOptimalCost << std::endl;
  ASSERT_NEAR(mProb1.cost, marginallyOptimalCost, 1e-4);

  // now v3 has been optimized, compute same marginal - should not be very different
  auto mProb2 =
      opt.computeMarginalProblem({.damping = 0}, {vars[0].index, vars[4].index, vars[5].index});
  std::cout << "[Marg2] H: " << mProb2.H.rows() << "x" << mProb2.H.cols()
            << "; b: " << mProb2.b.size() << "; c: " << mProb2.cost << std::endl;
  ASSERT_NEAR(mProb2.cost, marginallyOptimalCost, 1e-4);

  // now we shift (+2), and here variables will play the role of the selected vars above + base pt
  vector<Variable<Vec1>> vars2 = {{0}, {2}, {3.5}, {4.5}};

  // create opt problem with just this condensed factor which will be evaluated on
  // the results of: (w0 - w1), (w2 - w1), (w3 - w1)
  Optimizer opt2;
  opt2.addCondensedFactor(
      mProb2.H,
      mProb2.b,
      mProb2.cost,
      ProxyRelativeVecs<1>(
          &vars2[1], //
          {&vars2[0], &vars2[2], &vars2[3]}, //
          {Vec1{-2}, Vec1{1.5}, Vec1{2.5}} // linearization points
          ));

  ASSERT_TRUE(opt2.verifyJacobians());

  double condensedFactorCost = opt2.computeCost();
  std::cout << "'condensed factor' cost: " << condensedFactorCost << std::endl;
  ASSERT_NEAR(condensedFactorCost, marginallyOptimalCost, 1e-7);

  opt2.optimize();

  // `vars2` correspond to variables v0, v2, v4, v5 in the original problem
  ASSERT_NEAR(vars2[0].value[0] + 2.0, vars2[1].value[0], 1e-7);
  ASSERT_NEAR(vars2[1].value[0] + 2.0, vars2[2].value[0], 1e-7);
  ASSERT_NEAR(vars2[2].value[0] + 1.0, vars2[3].value[0], 1e-7);
}

template <int N>
static Eigen::Vector<double, N> randomVec(double maxAbs, std::mt19937& eng) {
  Eigen::Vector<double, N> ret;
  std::uniform_real_distribution<double> dist(-maxAbs, maxAbs);
  for (int i = 0; i < N; i++) {
    ret[i] = dist(eng);
  }
  return ret;
}

static Sophus::SE3d randomSE3(double maxTg, std::mt19937& eng) {
  return Sophus::SE3d::exp(randomVec<6>(maxTg, eng));
}

static double poseDistance(const Sophus::SE3d& p1, const Sophus::SE3d& p2) {
  return (p1 * p2.inverse()).log().norm();
}

static double vecDistance(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) {
  return (p1 - p2).norm();
}

TEST(TestCondensedFactor, PoseGraphPlain) {
  // generate Problem1's variables, both GT poses and perturbed poses, pose 0 is the identity
  static constexpr int nVars = 10;
  vector<Variable<Sophus::SE3d>> gt_T_rig_world(nVars), perturbed_T_rig_world(nVars);
  std::mt19937 eng(42);
  for (int q = 1; q < nVars; q++) {
    gt_T_rig_world[q].value = randomSE3(1.0, eng);
    perturbed_T_rig_world[q].value = randomSE3(0.05, eng) * gt_T_rig_world[q].value;
  }
  gt_T_rig_world[0].value = perturbed_T_rig_world[0].value = Sophus::SE3d();

  Optimizer opt;
  auto addConnection = [&](int i, int j) { // add an SE3 factor (boilerplate)
    Sophus::SE3d T_rig1_rig2 = gt_T_rig_world[i].value * gt_T_rig_world[j].value.inverse();
    opt.addFactor(
        [T_rig1_rig2 = T_rig1_rig2](
            const Sophus::SE3d& T_rig1_world,
            const Sophus::SE3d& T_rig2_world,
            Ref<Mat66>&& T_rig1_world_Jacobian,
            Ref<Mat66>&& T_rig2_world_Jacobian) -> Eigen::Vector<double, 6> {
          Sophus::SE3d errorAtRig1 = T_rig1_rig2 * T_rig2_world * T_rig1_world.inverse();
          Eigen::Vector<double, 6> logErrorAtRig1 = errorAtRig1.log();
          const Mat66 dLogError_dLeftPoseError = Sophus::SE3d::leftJacobianInverse(logErrorAtRig1);
          if (!isNull(T_rig1_world_Jacobian)) {
            T_rig1_world_Jacobian = -dLogError_dLeftPoseError * errorAtRig1.Adj();
          }
          if (!isNull(T_rig2_world_Jacobian)) {
            T_rig2_world_Jacobian = dLogError_dLeftPoseError * T_rig1_rig2.Adj();
          }
          return logErrorAtRig1;
        },
        perturbed_T_rig_world[i],
        perturbed_T_rig_world[j]);
  };

  // sparsely connect `perturbed_T_rig_world` vars, final problem is connected
  std::uniform_int_distribution<> oneInK(0, 4); // one in 5
  for (size_t i = 1; i < perturbed_T_rig_world.size(); i++) {
    std::uniform_int_distribution<> distrib(0, i - 1);
    int j = distrib(eng);
    addConnection(j, i); // connect to one of previous

    // randomly add some more connections
    for (size_t k = 0; k < i; k++) {
      if (k != j && (oneInK(eng) == 0)) {
        addConnection(k, i);
      }
    }
  }

  // pose0 is set to constant
  perturbed_T_rig_world[0].setConstant(true);

  // for reference, compute the cost
  double baseCost = opt.computeCost();
  std::cout << "Base cost: " << baseCost << std::endl;

  // select variables in Prob1 which will stay. define some utilities
  std::vector<int64_t> survivingVariables{3, 7, 9};
  auto updatedIndicesOfAllSurvivingVariables = [&]() -> std::vector<int64_t> {
    std::vector<int64_t> ret;
    for (int64_t i : survivingVariables) {
      ret.push_back(perturbed_T_rig_world[i].index);
    }
    return ret;
  };

  // make sure the variables we specify have an index (TODO: simplify)
  opt.registerAllVariables();
  auto mProb1 = opt.computeMarginalProblem(
      {.damping = 0}, //
      updatedIndicesOfAllSurvivingVariables() //
  );

  std::cout << "[Marg1] H: " << mProb1.H.rows() << "x" << mProb1.H.cols()
            << "; b: " << mProb1.b.size() << "; c: " << mProb1.cost << std::endl;

  // check what the cost is when {s_i} variables are fixed, and other vars optimized accordingly
  opt.unregisterAllVariables();
  for (auto& s : survivingVariables) {
    perturbed_T_rig_world[s].setConstant(true);
  }

  ASSERT_TRUE(opt.verifyJacobians());

  opt.optimize();
  double marginallyOptimalCost = opt.computeCost();
  std::cout << "'marginally optimal' cost:" << marginallyOptimalCost << std::endl;
  ASSERT_NEAR(mProb1.cost, marginallyOptimalCost, 1.5e-4);

  // recompute the marginal problem, the {m_i} have now been optimized, but results should be close
  opt.unregisterAllVariables();
  for (auto& s : survivingVariables) {
    perturbed_T_rig_world[s].setConstant(false);
  }
  opt.registerAllVariables();
  auto mProb2 = opt.computeMarginalProblem(
      {.damping = 0}, //
      updatedIndicesOfAllSurvivingVariables() //
  );
  std::cout << "[Marg2] H: " << mProb2.H.rows() << "x" << mProb2.H.cols()
            << "; b: " << mProb2.b.size() << "; c: " << mProb2.cost << std::endl;
  ASSERT_NEAR(mProb2.cost, marginallyOptimalCost, 1e-4);

  // We now use the marginal problem in Problem2
  Sophus::SE3d T_world1_world2 = randomSE3(1.0, eng);
  vector<Variable<Sophus::SE3d>> alt_T_rig_world{T_world1_world2};
  for (int64_t s : survivingVariables) {
    alt_T_rig_world.push_back(perturbed_T_rig_world[s].value * T_world1_world2);
  }

  Optimizer opt2;
  std::vector<Variable<Sophus::SE3d>*> otherPoseVars;
  std::vector<Sophus::SE3d> poseLinPts;
  for (size_t i = 1; i < alt_T_rig_world.size(); i++) {
    otherPoseVars.push_back(&alt_T_rig_world[i]);
    poseLinPts.push_back(perturbed_T_rig_world[survivingVariables[i - 1]].value);
  }
  opt2.addCondensedFactor(
      mProb2.H,
      mProb2.b,
      mProb2.cost,
      ProxyRelativePoses(
          &alt_T_rig_world[0], // pose1
          otherPoseVars, // (N-1) x poseI variables, for i >= 2
          poseLinPts // linearization points, ie relative poses from Problem1
          ));

  // this can be used to verify that jacobians in "Proxy" functions are correct
  ASSERT_TRUE(opt2.verifyJacobians());

  // the cost should match the "marginally optimal" cost
  double condensedFactorCost = opt2.computeCost();
  std::cout << "'condensed factor' cost: " << condensedFactorCost << std::endl;
  ASSERT_NEAR(condensedFactorCost, marginallyOptimalCost, 1e-5);

  opt2.optimize();

  for (int64_t i = 0; i < survivingVariables.size(); i++) {
    int64_t s = survivingVariables[i]; // index in Problem1
    ASSERT_NEAR(
        poseDistance(
            alt_T_rig_world[1 + i].value * alt_T_rig_world[0].value.inverse(),
            gt_T_rig_world[s].value),
        0,
        7e-4);
  }
}

// more complex example with velocities
TEST(TestCondensedFactor, PoseGraphWithVelocities) {
  // generate Problem1's variables, both GT poses and perturbed poses, pose 0 is the identity
  static constexpr int nVars = 10;
  vector<Variable<Sophus::SE3d>> gt_T_rig_world(nVars), perturbed_T_rig_world(nVars);
  vector<Variable<Eigen::Vector3d>> gt_vel_world(nVars), perturbed_vel_world(nVars);
  std::mt19937 eng(42);
  for (int q = 1; q < nVars; q++) {
    gt_T_rig_world[q].value = randomSE3(1.0, eng);
    perturbed_T_rig_world[q].value = randomSE3(0.05, eng) * gt_T_rig_world[q].value;
  }
  gt_T_rig_world[0].value = perturbed_T_rig_world[0].value = Sophus::SE3d();
  for (int q = 0; q < nVars; q++) {
    gt_vel_world[q].value = randomVec<3>(1.0, eng);
    perturbed_vel_world[q].value = randomVec<3>(0.05, eng) + gt_vel_world[q].value;
  }

  Optimizer opt;
  auto addConnection = [&](int i, int j) { // add an SE3 factor (boilerplate)
    Sophus::SE3d T_rig1_rig2 = gt_T_rig_world[i].value * gt_T_rig_world[j].value.inverse();
    opt.addFactor(
        [T_rig1_rig2 = T_rig1_rig2](
            const Sophus::SE3d& T_rig1_world,
            const Sophus::SE3d& T_rig2_world,
            Ref<Mat66>&& T_rig1_world_Jacobian,
            Ref<Mat66>&& T_rig2_world_Jacobian) -> Eigen::Vector<double, 6> {
          Sophus::SE3d errorAtRig1 = T_rig1_rig2 * T_rig2_world * T_rig1_world.inverse();
          Eigen::Vector<double, 6> logErrorAtRig1 = errorAtRig1.log();
          const Mat66 dLogError_dLeftPoseError = Sophus::SE3d::leftJacobianInverse(logErrorAtRig1);
          if (!isNull(T_rig1_world_Jacobian)) {
            T_rig1_world_Jacobian = -dLogError_dLeftPoseError * errorAtRig1.Adj();
          }
          if (!isNull(T_rig2_world_Jacobian)) {
            T_rig2_world_Jacobian = dLogError_dLeftPoseError * T_rig1_rig2.Adj();
          }
          return logErrorAtRig1;
        },
        perturbed_T_rig_world[i],
        perturbed_T_rig_world[j]);
  };

  // sparsely connect `perturbed_T_rig_world` vars, final problem is connected
  std::uniform_int_distribution<> oneInK(0, 4); // one in 5
  for (size_t i = 1; i < perturbed_T_rig_world.size(); i++) {
    std::uniform_int_distribution<> distrib(0, i - 1);
    int j = distrib(eng);
    addConnection(j, i); // connect to one of previous

    // randomly add some more connections
    for (size_t k = 0; k < i; k++) {
      if (k != j && (oneInK(eng) == 0)) {
        addConnection(k, i);
      }
    }
  }

  // add simple priors on all velocities
  for (size_t i = 0; i < nVars; i++) {
    opt.addFactor(
        [gt_vel_w = gt_vel_world[i].value](
            const Eigen::Vector3d& vel_world,
            Ref<Eigen::Matrix3d>&& vel_world_Jacobian) -> Eigen::Vector3d {
          if (!isNull(vel_world_Jacobian)) {
            vel_world_Jacobian.setIdentity();
          }
          return vel_world - gt_vel_w;
        },
        perturbed_vel_world[i]);
  }

  // pose0 is set to constant
  perturbed_T_rig_world[0].setConstant(true);

  // for reference, compute the cost
  double baseCost = opt.computeCost();
  std::cout << "Base cost: " << baseCost << std::endl;

  // select variables in Prob1 which will stay. define some utilities
  std::vector<int64_t> survivingVariables{3, 7, 9};
  auto updatedIndicesOfAllSurvivingVariables = [&]() -> std::vector<int64_t> {
    std::vector<int64_t> ret;
    for (int64_t i : survivingVariables) {
      ret.push_back(perturbed_T_rig_world[i].index);
    }
    ret.push_back(perturbed_vel_world[0].index); // vel of pose1
    for (int64_t i : survivingVariables) {
      ret.push_back(perturbed_vel_world[i].index); // other vels
    }
    return ret;
  };
  auto survivingVariablesSetConstant = [&](bool constant) {
    for (int64_t i : survivingVariables) {
      perturbed_T_rig_world[i].setConstant(constant);
    }
    perturbed_vel_world[0].setConstant(constant); // vel of pose1
    for (int64_t i : survivingVariables) {
      perturbed_vel_world[i].setConstant(constant); // other vels
    }
  };

  // make sure the variables we specify have an index (TODO: simplify)
  opt.registerAllVariables();
  auto mProb1 = opt.computeMarginalProblem(
      {.damping = 0}, //
      updatedIndicesOfAllSurvivingVariables() //
  );

  std::cout << "[Marg1] H: " << mProb1.H.rows() << "x" << mProb1.H.cols()
            << "; b: " << mProb1.b.size() << "; c: " << mProb1.cost << std::endl;

  // check what the cost is when {s_i} variables are fixed, and other vars optimized accordingly
  opt.unregisterAllVariables();
  survivingVariablesSetConstant(true);

  ASSERT_TRUE(opt.verifyJacobians());

  opt.optimize();
  double marginallyOptimalCost = opt.computeCost();
  std::cout << "'marginally optimal' cost:" << marginallyOptimalCost << std::endl;
  ASSERT_NEAR(mProb1.cost, marginallyOptimalCost, 1e-3);

  // recompute the marginal problem, the {m_i} have now been optimized, but results should be close
  opt.unregisterAllVariables();
  survivingVariablesSetConstant(false);
  opt.registerAllVariables();
  auto mProb2 = opt.computeMarginalProblem(
      {.damping = 0}, //
      updatedIndicesOfAllSurvivingVariables() //
  );
  std::cout << "[Marg2] H: " << mProb2.H.rows() << "x" << mProb2.H.cols()
            << "; b: " << mProb2.b.size() << "; c: " << mProb2.cost << std::endl;
  ASSERT_NEAR(mProb2.cost, marginallyOptimalCost, 1e-4);

  // We now use the marginal problem in Problem2
  Sophus::SE3d T_world1_world2 = randomSE3(1.0, eng);
  vector<Variable<Sophus::SE3d>> alt_T_rig_world{T_world1_world2};
  vector<Variable<Eigen::Vector3d>> alt_vel_world{
      T_world1_world2.so3().inverse() * perturbed_vel_world[0].value};
  for (int64_t s : survivingVariables) {
    alt_T_rig_world.push_back(perturbed_T_rig_world[s].value * T_world1_world2);
    alt_vel_world.push_back(T_world1_world2.so3().inverse() * perturbed_vel_world[s].value);
  }

  Optimizer opt2;
  std::vector<Variable<Sophus::SE3d>*> otherPoseVars;
  std::vector<Sophus::SE3d> poseLinPts;
  std::vector<Variable<Eigen::Vector3d>*> velVars{&alt_vel_world[0]}; // vel of pose0
  std::vector<Eigen::Vector3d> velLinPts{perturbed_vel_world[0].value};
  for (size_t i = 1; i < alt_T_rig_world.size(); i++) {
    otherPoseVars.push_back(&alt_T_rig_world[i]);
    poseLinPts.push_back(perturbed_T_rig_world[survivingVariables[i - 1]].value);
    velVars.push_back(&alt_vel_world[i]);
    velLinPts.push_back(perturbed_vel_world[survivingVariables[i - 1]].value);
  }
  opt2.addCondensedFactor(
      mProb2.H,
      mProb2.b,
      mProb2.cost,
      ProxyRelativePoses(
          &alt_T_rig_world[0], // pose1
          otherPoseVars, // (N-1) x poseI variables, for i >= 2
          poseLinPts // linearization points, ie relative poses from Problem1
          ),
      ProxyTransformedVelocities(
          &alt_T_rig_world[0], // pose1
          velVars, // N x velI variables, for i >= 1
          velLinPts // linearization points, ie relative velocities from Problem1
          ));

  // this can be used to verify that jacobians in "Proxy" functions are correct
  ASSERT_TRUE(opt2.verifyJacobians());

  // the cost should match the "marginally optimal" cost
  double condensedFactorCost = opt2.computeCost();
  std::cout << "'condensed factor' cost: " << condensedFactorCost << std::endl;
  ASSERT_NEAR(condensedFactorCost, marginallyOptimalCost, 1e-5);

  opt2.optimize();

  // verify poses optimized using condensed factor
  for (int64_t i = 0; i < survivingVariables.size(); i++) {
    int64_t s = survivingVariables[i]; // index in Problem1
    ASSERT_NEAR(
        poseDistance(
            alt_T_rig_world[1 + i].value * alt_T_rig_world[0].value.inverse(),
            gt_T_rig_world[s].value),
        0,
        8e-4);
  }

  // verify velocities optimized using condensed factor
  for (int64_t i = 0; i < velVars.size(); i++) {
    int64_t s = (i == 0) ? 0 : survivingVariables[i - 1]; // index in Problem1
    ASSERT_NEAR(
        vecDistance(alt_T_rig_world[0].value.so3() * alt_vel_world[i].value, gt_vel_world[s].value),
        0,
        8e-7);
  }
}

// more complex example with velocities
TEST(TestCondensedFactor, PoseGraphWithVelocitiesAndGravity) {
  // generate Problem1's variables, both GT poses and perturbed poses, pose 0 is the identity
  static constexpr int nVars = 10;
  Variable<S2> gt_gravity = S2{.radius = 9.81, .vec = {0, 0, -9.81}};
  vector<Variable<Sophus::SE3d>> gt_T_rig_world(nVars), perturbed_T_rig_world(nVars);
  vector<Variable<Eigen::Vector3d>> gt_vel_world(nVars), perturbed_vel_world(nVars);
  std::mt19937 eng(42);
  for (int q = 1; q < nVars; q++) {
    gt_T_rig_world[q].value = randomSE3(1.0, eng);
    perturbed_T_rig_world[q].value = randomSE3(0.05, eng) * gt_T_rig_world[q].value;
  }
  gt_T_rig_world[0].value = perturbed_T_rig_world[0].value = Sophus::SE3d();
  for (int q = 0; q < nVars; q++) {
    gt_vel_world[q].value = randomVec<3>(1.0, eng);
    perturbed_vel_world[q].value = randomVec<3>(0.05, eng) + gt_vel_world[q].value;
  }
  Variable<S2> perturbed_gravity = gt_gravity.value;
  VarSpec<S2>::applyBoxPlus(perturbed_gravity.value, randomVec<2>(0.03, eng));

  Optimizer opt;
  auto addConnection = [&](int i, int j) { // add an SE3 factor (boilerplate)
    Sophus::SE3d T_rig1_rig2 = gt_T_rig_world[i].value * gt_T_rig_world[j].value.inverse();
    opt.addFactor(
        [T_rig1_rig2 = T_rig1_rig2](
            const Sophus::SE3d& T_rig1_world,
            const Sophus::SE3d& T_rig2_world,
            Ref<Mat66>&& T_rig1_world_Jacobian,
            Ref<Mat66>&& T_rig2_world_Jacobian) -> Eigen::Vector<double, 6> {
          Sophus::SE3d errorAtRig1 = T_rig1_rig2 * T_rig2_world * T_rig1_world.inverse();
          Eigen::Vector<double, 6> logErrorAtRig1 = errorAtRig1.log();
          const Mat66 dLogError_dLeftPoseError = Sophus::SE3d::leftJacobianInverse(logErrorAtRig1);
          if (!isNull(T_rig1_world_Jacobian)) {
            T_rig1_world_Jacobian = -dLogError_dLeftPoseError * errorAtRig1.Adj();
          }
          if (!isNull(T_rig2_world_Jacobian)) {
            T_rig2_world_Jacobian = dLogError_dLeftPoseError * T_rig1_rig2.Adj();
          }
          return logErrorAtRig1;
        },
        perturbed_T_rig_world[i],
        perturbed_T_rig_world[j]);
  };

  // sparsely connect `perturbed_T_rig_world` vars, final problem is connected
  std::uniform_int_distribution<> oneInK(0, 4); // one in 5
  for (size_t i = 1; i < perturbed_T_rig_world.size(); i++) {
    std::uniform_int_distribution<> distrib(0, i - 1);
    int j = distrib(eng);
    addConnection(j, i); // connect to one of previous

    // randomly add some more connections
    for (size_t k = 0; k < i; k++) {
      if (k != j && (oneInK(eng) == 0)) {
        addConnection(k, i);
      }
    }
  }

  // add simple gravity priors
  for (size_t i = 0; i < nVars; i++) {
    S2 gt_local_gravity = gt_T_rig_world[i].value.so3() * gt_gravity.value;
    opt.addFactor(
        [gt_local_gravity](
            const S2& world_gravity,
            const Sophus::SE3d& T_rig_world,
            Ref<Eigen::Matrix<double, 3, 2>>&& gravity_Jacobian,
            Ref<Eigen::Matrix<double, 3, 6>>&& T_rig_world_Jacobian) -> Eigen::Vector3d {
          S2 local_gravity = T_rig_world.so3() * world_gravity;
          if (!isNull(gravity_Jacobian)) {
            gravity_Jacobian =
                T_rig_world.so3().matrix() * S2::ortho(world_gravity.vec).transpose();
          }
          if (!isNull(T_rig_world_Jacobian)) {
            T_rig_world_Jacobian << Eigen::Matrix<double, 3, 3>::Zero(),
                Sophus::SO3d::hat(-local_gravity.vec);
          }
          return local_gravity.vec - gt_local_gravity.vec;
        },
        perturbed_gravity,
        perturbed_T_rig_world[i]);
  }

  // add simple priors on all velocities
  for (size_t i = 0; i < nVars; i++) {
    opt.addFactor(
        [gt_vel_w = gt_vel_world[i].value](
            const Eigen::Vector3d& vel_world,
            Ref<Eigen::Matrix3d>&& vel_world_Jacobian) -> Eigen::Vector3d {
          if (!isNull(vel_world_Jacobian)) {
            vel_world_Jacobian.setIdentity();
          }
          return vel_world - gt_vel_w;
        },
        perturbed_vel_world[i]);
  }

  // pose0 is set to constant
  perturbed_T_rig_world[0].setConstant(true);

  // for reference, compute the cost
  double baseCost = opt.computeCost();
  std::cout << "Base cost: " << baseCost << std::endl;

  // select variables in Prob1 which will stay. define some utilities
  std::vector<int64_t> survivingVariables{3, 7, 9};
  auto updatedIndicesOfAllSurvivingVariables = [&]() -> std::vector<int64_t> {
    std::vector<int64_t> ret;
    ret.push_back(perturbed_gravity.index);
    for (int64_t i : survivingVariables) {
      ret.push_back(perturbed_T_rig_world[i].index);
    }
    ret.push_back(perturbed_vel_world[0].index); // vel of pose1
    for (int64_t i : survivingVariables) {
      ret.push_back(perturbed_vel_world[i].index); // other vels
    }
    return ret;
  };
  auto survivingVariablesSetConstant = [&](bool constant) {
    perturbed_gravity.setConstant(constant);
    for (int64_t i : survivingVariables) {
      perturbed_T_rig_world[i].setConstant(constant);
    }
    perturbed_vel_world[0].setConstant(constant); // vel of pose1
    for (int64_t i : survivingVariables) {
      perturbed_vel_world[i].setConstant(constant); // other vels
    }
  };

  // make sure the variables we specify have an index (TODO: simplify)
  opt.registerAllVariables();
  auto mProb1 = opt.computeMarginalProblem(
      {.damping = 0}, //
      updatedIndicesOfAllSurvivingVariables() //
  );

  std::cout << "[Marg1] H: " << mProb1.H.rows() << "x" << mProb1.H.cols()
            << "; b: " << mProb1.b.size() << "; c: " << mProb1.cost << std::endl;

  // check what the cost is when {s_i} variables are fixed, and other vars optimized accordingly
  opt.unregisterAllVariables();
  survivingVariablesSetConstant(true);

  ASSERT_TRUE(opt.verifyJacobians());

  opt.optimize();
  double marginallyOptimalCost = opt.computeCost();
  std::cout << "'marginally optimal' cost:" << marginallyOptimalCost << std::endl;
  ASSERT_NEAR(mProb1.cost, marginallyOptimalCost, 1e-3);

  // recompute the marginal problem, the {m_i} have now been optimized, but results should be close
  opt.unregisterAllVariables();
  survivingVariablesSetConstant(false);
  opt.registerAllVariables();
  auto mProb2 = opt.computeMarginalProblem(
      {.damping = 0}, //
      updatedIndicesOfAllSurvivingVariables() //
  );
  std::cout << "[Marg2] H: " << mProb2.H.rows() << "x" << mProb2.H.cols()
            << "; b: " << mProb2.b.size() << "; c: " << mProb2.cost << std::endl;
  ASSERT_NEAR(mProb2.cost, marginallyOptimalCost, 1e-4);

  // We now use the marginal problem in Problem2
  Sophus::SE3d T_world1_world2 = randomSE3(1.0, eng);
  vector<Variable<Sophus::SE3d>> alt_T_rig_world{T_world1_world2};
  vector<Variable<Eigen::Vector3d>> alt_vel_world{
      T_world1_world2.so3().inverse() * perturbed_vel_world[0].value};
  for (int64_t s : survivingVariables) {
    alt_T_rig_world.push_back(perturbed_T_rig_world[s].value * T_world1_world2);
    alt_vel_world.push_back(T_world1_world2.so3().inverse() * perturbed_vel_world[s].value);
  }

  Optimizer opt2;
  Variable<S2> alt_gravity_var = T_world1_world2.so3().inverse() * perturbed_gravity.value;
  S2 gravityLinPt = perturbed_gravity.value;
  std::vector<Variable<Sophus::SE3d>*> otherPoseVars;
  std::vector<Sophus::SE3d> poseLinPts;
  std::vector<Variable<Eigen::Vector3d>*> velVars{&alt_vel_world[0]}; // vel of pose0
  std::vector<Eigen::Vector3d> velLinPts{perturbed_vel_world[0].value};
  for (size_t i = 1; i < alt_T_rig_world.size(); i++) {
    otherPoseVars.push_back(&alt_T_rig_world[i]);
    poseLinPts.push_back(perturbed_T_rig_world[survivingVariables[i - 1]].value);
    velVars.push_back(&alt_vel_world[i]);
    velLinPts.push_back(perturbed_vel_world[survivingVariables[i - 1]].value);
  }
  opt2.addCondensedFactor(
      mProb2.H,
      mProb2.b,
      mProb2.cost,
      ProxyS2(
          &alt_T_rig_world[0], // pose1
          {&alt_gravity_var},
          {gravityLinPt}),
      ProxyRelativePoses(
          &alt_T_rig_world[0], // pose1
          otherPoseVars, // (N-1) x poseI variables, for i >= 2
          poseLinPts // linearization points, ie relative poses from Problem1
          ),
      ProxyTransformedVelocities(
          &alt_T_rig_world[0], // pose1
          velVars, // N x velI variables, for i >= 1
          velLinPts // linearization points, ie relative velocities from Problem1
          ));

  // this can be used to verify that jacobians in "Proxy" functions are correct
  ASSERT_TRUE(opt2.verifyJacobians());

  // the cost should match the "marginally optimal" cost
  double condensedFactorCost = opt2.computeCost();
  std::cout << "'condensed factor' cost: " << condensedFactorCost << std::endl;
  ASSERT_NEAR(condensedFactorCost, marginallyOptimalCost, 1e-5);

  opt2.optimize();

  // verify poses optimized using condensed factor
  for (int64_t i = 0; i < survivingVariables.size(); i++) {
    int64_t s = survivingVariables[i]; // index in Problem1
    ASSERT_NEAR(
        poseDistance(
            alt_T_rig_world[1 + i].value * alt_T_rig_world[0].value.inverse(),
            gt_T_rig_world[s].value),
        0,
        1.2e-3);
  }

  // verify velocities optimized using condensed factor
  for (int64_t i = 0; i < velVars.size(); i++) {
    int64_t s = (i == 0) ? 0 : survivingVariables[i - 1]; // index in Problem1
    ASSERT_NEAR(
        vecDistance(alt_T_rig_world[0].value.so3() * alt_vel_world[i].value, gt_vel_world[s].value),
        0,
        8e-7);
  }

  // verify gravity estimation
  ASSERT_NEAR(
      vecDistance(
          (alt_T_rig_world[0].value.so3() * alt_gravity_var.value).vec, gt_gravity.value.vec),
      0,
      2e-5);
}
