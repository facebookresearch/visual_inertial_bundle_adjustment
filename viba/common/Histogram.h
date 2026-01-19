/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <viba/common/Utf8Symbols.h>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

class Histogram {
 public:
  struct ShowSettings {
    int width = 150;
    int precision = 0;
    std::string symbol = utf8symbols::kNeutralFace;
    int symbolWidth = 2;
  };

  explicit Histogram(const std::vector<double>& buckets);

  Histogram(double min, double max, int num_buckets);

  void split_bucket(int num, int split_size);

  void stat(double num);

  Histogram clone() const {
    return Histogram(_buckets);
  }

  void collectFrom(const Histogram& that);

  std::string show(
      const ShowSettings& /*settings*/ = {
          .width = 150,
          .precision = 0,
          .symbol = utf8symbols::kNeutralFace,
          .symbolWidth = 2}) const;

  const std::vector<int64_t>& getStats() const {
    return _stat;
  }

 private:
  std::vector<double> _buckets;
  std::vector<int64_t> _stat;
};
