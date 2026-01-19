/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <small_thing/Optimizer.h>

namespace visual_inertial_ba {

struct Histograms {
  enum Color {
    None = 0,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    ColorsStart = Red,
    ColorsEnd = Cyan,
  };

  explicit Histograms(small_thing::Optimizer& opt) : opt(opt) {}

  void show() const;

  void showHistogramsVisual(
      const std::vector<small_thing::FactorStoreBase*>& allVisualFactors,
      small_thing::FactorStoreBase* baseMapVisualFactors) const;

  void showHistogramsInertial(
      small_thing::FactorStoreBase* inertialFactors,
      std::vector<small_thing::FactorStoreBase*> allSecondaryInertialFactors) const;

  void showHistogramsCalib(
      bool isFactoryCalibPriors,
      small_thing::FactorStoreBase* imuCalibRWFactors,
      std::vector<small_thing::FactorStoreBase*> allCamIntrRWFactors,
      small_thing::FactorStoreBase* imuExtrRWFactors,
      small_thing::FactorStoreBase* camExtrRWFactors) const;

  void showHistogramsOmegaPriors(
      std::vector<small_thing::FactorStoreBase*> omegaPriorFactors) const;

  small_thing::Optimizer& opt;

  bool showVisual = true;
  bool showPixelErrors = true; // image-distance pixel reproj errors
  bool showInertial = true;
  bool separateSecondaryInertial = true;
  bool showRotVelPos = true; // separate histograms for rot/vel/pos
  bool showRandomWalks = true;
  bool showFactoryCalibPriors = true;
  bool showAggregateCalibFactors = false; // one histogram for all rw/facprio factors
  bool showOmegaPriors = true;

  // (optional) function classifying the factors, used to print
  // separate histograms per-recording, or to separate tracking/global points
  using FactorRefToGroupIndex = std::function<int(small_thing::FactorStoreBase*, int64_t)>;
  using GroupIndexToColor = std::function<Color(int)>;

  struct HistSetting {
    FactorRefToGroupIndex factorToGroup;
    std::vector<std::string> groupLabel = {std::string()};
    GroupIndexToColor groupCol = [](int) { return None; };
  };

  HistSetting visual = {.groupCol = [](int) { return Green; }};
  HistSetting baseMap = {.groupCol = [](int) { return Magenta; }};
  HistSetting inertial = {.groupCol = [](int) { return Red; }};
  HistSetting secondaryInertial = {.groupCol = [](int) { return Magenta; }};
  HistSetting rwImuCalib = {.groupCol = [](int) { return Yellow; }};
  HistSetting rwCamIntr = {.groupCol = [](int) { return Yellow; }};
  HistSetting rwImuExtr = {.groupCol = [](int) { return Yellow; }};
  HistSetting rwCamExtr = {.groupCol = [](int) { return Yellow; }};
  HistSetting omegaPriors = {.groupCol = [](int) { return Cyan; }};
  HistSetting fpImuCalib = {.groupCol = [](int) { return Magenta; }};
  HistSetting fpCamIntr = {.groupCol = [](int) { return Magenta; }};
  HistSetting fpImuExtr = {.groupCol = [](int) { return Magenta; }};
  HistSetting fpCamExtr = {.groupCol = [](int) { return Magenta; }};
};

} // namespace visual_inertial_ba
