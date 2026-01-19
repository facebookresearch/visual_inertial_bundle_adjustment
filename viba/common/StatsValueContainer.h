/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

class StatsValueContainer {
 public:
  StatsValueContainer();

  // Add a new value to the container
  void add(double value) const;

  // add all values in vector
  void addValues(const std::vector<double>& vals);

  // Return the mean of the values
  double mean() const;

  // Return the root mean square error (RMSE) of the values
  double rmse() const;

  // Return the p50 (median) value
  double p50() const;

  // Return the p90 value
  double p90() const;

  // Return the pX percentile value (X in [0, 100])
  double pX(double percentile) const;

  double min() const;

  double max() const;

  // Return the number of values
  size_t size() const;

  // return values
  const std::vector<double>& values() const {
    return values_;
  }

 private:
  void sortIfNeeded() const;

  mutable std::vector<double> values_;
  mutable bool sorted_;
};
