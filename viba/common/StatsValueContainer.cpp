/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "StatsValueContainer.h"

StatsValueContainer::StatsValueContainer() : sorted_(true) {}

void StatsValueContainer::add(double value) const {
  values_.push_back(value);
  sorted_ = false;
}

void StatsValueContainer::addValues(const std::vector<double>& vals) {
  values_.insert(values_.end(), vals.begin(), vals.end());
  sorted_ = false;
}

double StatsValueContainer::mean() const {
  if (values_.empty()) {
    throw std::runtime_error("No values in container");
  }
  double sum = 0.0;
  for (double v : values_) {
    sum += v;
  }
  return sum / values_.size();
}

double StatsValueContainer::rmse() const {
  if (values_.empty()) {
    throw std::runtime_error("No values in container");
  }
  double sum_sq = 0.0;
  for (double v : values_) {
    sum_sq += v * v;
  }
  return std::sqrt(sum_sq / values_.size());
}

double StatsValueContainer::p50() const {
  return pX(50.0);
}

double StatsValueContainer::p90() const {
  return pX(90.0);
}

double StatsValueContainer::pX(double percentile) const {
  if (values_.empty()) {
    throw std::runtime_error("No values in container");
  }
  if (percentile < 0.0 || percentile > 100.0) {
    throw std::invalid_argument("Percentile must be in [0, 100]");
  }
  sortIfNeeded();
  // Nearest-rank method
  double pos = (percentile / 100.0) * (values_.size() - 1);
  size_t idx_below = static_cast<size_t>(std::floor(pos));
  size_t idx_above = static_cast<size_t>(std::ceil(pos));
  if (idx_below == idx_above) {
    return values_[idx_below];
  } else {
    double weight_above = pos - idx_below;
    double weight_below = 1.0 - weight_above;
    return values_[idx_below] * weight_below + values_[idx_above] * weight_above;
  }
}

double StatsValueContainer::min() const {
  if (values_.empty()) {
    throw std::runtime_error("No values in container");
  }
  sortIfNeeded();
  return values_[0];
}

double StatsValueContainer::max() const {
  if (values_.empty()) {
    throw std::runtime_error("No values in container");
  }
  sortIfNeeded();
  return values_.back();
}

size_t StatsValueContainer::size() const {
  return values_.size();
}

void StatsValueContainer::sortIfNeeded() const {
  if (!sorted_) {
    std::sort(values_.begin(), values_.end());
    sorted_ = true;
  }
}
