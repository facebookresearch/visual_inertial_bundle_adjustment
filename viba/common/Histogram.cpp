/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
#include <viba/common/Histogram.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <sstream>

using namespace std;

// find biggest index i such that buckets[i] <= val
static int find_index(const std::vector<double>& buckets, const double val) {
  const auto upper = std::upper_bound(buckets.begin(), buckets.end(), val);
  return std::distance(buckets.begin(), upper);
}

Histogram::Histogram(const std::vector<double>& buckets) : _buckets(buckets) {
  _stat.resize(_buckets.size() + 1);
}

Histogram::Histogram(double bmin, double bmax, int num_buckets) {
  for (int i = 0; i <= num_buckets; i++) {
    double v = bmin + (bmax - bmin) * i / double(num_buckets);
    _buckets.push_back(v);
  }
  _stat.resize(_buckets.size() + 1);
}

void Histogram::collectFrom(const Histogram& that) {
  if (that._stat.size() != _stat.size()) {
    throw std::runtime_error("histogram size doesn't match");
  }
  for (size_t i = 0; i < _stat.size(); i++) {
    _stat[i] += that._stat[i];
  }
}

void Histogram::split_bucket(int num, int split_size) {
  double low = _buckets[num], hi = _buckets[num + 1];
  for (int i = split_size - 1; i > 0; i--) {
    _buckets.insert(_buckets.begin() + (num + 1), low + (hi - low) * i / split_size);
  }
  _stat.resize(_buckets.size() + 1);
}

void Histogram::stat(double num) {
  int i = find_index(_buckets, num);
  _stat[i] += 1;
}

string Histogram::show(const ShowSettings& settings) const {
  int a = 0, b = _stat.size();
  while (a < (int)_stat.size() && _stat[a] == 0) {
    a++;
  }
  while (b > 0 && _stat[b - 1] == 0) {
    b--;
  }
  int64_t maxlabellen = 0;
  int64_t maxstat = 0;
  vector<string> labels;
  vector<int> labelLengths;
  for (int i = a; i < b; i++) {
    std::stringstream stream;
    int loff = 0;
    if (i > 0) {
      stream << std::fixed << std::setprecision(settings.precision) << _buckets[i - 1];
    } else {
      stream << "-" << utf8symbols::kInfinity;
      loff += strlen(utf8symbols::kInfinity) - 1;
    }
    stream << utf8symbols::kEllipsis;
    loff += strlen(utf8symbols::kEllipsis) - 1;
    if (i < (int)_buckets.size()) {
      stream << std::fixed << std::setprecision(settings.precision) << _buckets[i];
    } else {
      stream << utf8symbols::kInfinity;
      loff += strlen(utf8symbols::kInfinity) - 1;
    }

    string s = stream.str();
    labels.push_back(s);
    labelLengths.push_back(s.length() - loff);
    maxlabellen = max((size_t)maxlabellen, s.length() - loff);
    maxstat = max(maxstat, _stat[i]);
  }

  int histogram_width = settings.width - maxlabellen - 2;

  std::stringstream stream;
  for (int i = a; i < b; i++) {
    stream << setfill(' ') << setw(maxlabellen - labelLengths[i - a]) << "" << labels[i - a]
           << ": ";
    int line_len = int((_stat[i] * (double)histogram_width) / (double)maxstat + 0.5);
    if (line_len * 3 > histogram_width * 2) {
      string s3 = to_string(_stat[i]);
      int rem = (line_len - s3.length() - 2) / settings.symbolWidth;
      int remh1 = rem / 2;
      for (int q = 0; q < remh1; q++) {
        stream << settings.symbol;
      }
      stream << " " << s3 << " ";
      for (int q = 0; q < rem - remh1; q++) {
        stream << settings.symbol;
      }
    } else {
      for (int q = 0; q < line_len / settings.symbolWidth; q++) {
        stream << settings.symbol;
      }
      stream << "  " << _stat[i];
    }
    stream << endl;
  }
  return stream.str();
}
