/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <baspacho/baspacho/DebugMacros.h>
#include <cmath>
#include <tuple>

namespace small_thing {

struct Loss {};

struct TrivialLoss : Loss {
  explicit TrivialLoss(double /* a */ = 0) {}

  void setSize(double /* a */) {}

  static inline double val(double s) {
    return s;
  }

  static inline double der(double s) {
    return 1.0;
  }

  static inline std::pair<double, double> jet2(double s) {
    return {s, 1.0};
  }

  static inline std::tuple<double, double, double> jet3(double s) {
    return {s, 1.0, 0.0};
  }
};

struct L1Loss : Loss {
  explicit L1Loss(double /* a */ = 0) {}

  void setSize(double /* a */) {}

  static inline double val(double s) {
    return std::sqrt(s);
  }

  static inline double der(double s) {
    return 0.5 / std::sqrt(s);
  }

  static inline std::pair<double, double> jet2(double s) {
    const double r = std::sqrt(s);
    return {r, 0.5 / r};
  }

  static inline std::tuple<double, double, double> jet3(double s) {
    const double r = std::sqrt(s);
    return {r, 0.5 / r, -0.25 / (s * r)};
  }
};

struct HuberLoss : Loss {
  double a, b;
  explicit HuberLoss(double a) : a(a), b(a * a) {
    BASPACHO_CHECK_GT(a, 0.0);
  }

  void setSize(double _a) {
    BASPACHO_CHECK_GT(_a, 0.0);
    a = _a;
    b = _a * _a;
  }

  inline double val(double s) const {
    if (s > b) {
      const double r = std::sqrt(s);
      return 2.0 * a * r - b;
    } else {
      return s;
    }
  }

  inline double der(double s) const {
    if (s > b) {
      const double r = std::sqrt(s);
      return a / r;
    } else {
      return 1.0;
    }
  }

  inline std::pair<double, double> jet2(double s) const {
    if (s > b) {
      const double r = std::sqrt(s);
      const double d = a / r;
      return {2.0 * a * r - b, d};
    } else {
      return {s, 1.0};
    }
  }

  inline std::tuple<double, double, double> jet3(double s) const {
    if (s > b) {
      const double r = std::sqrt(s);
      const double d = a / r;
      return {2.0 * a * r - b, d, -d / (2.0 * s)};
    } else {
      return {s, 1.0, 0.0};
    }
  }
};

struct HuberLossWithCutoff : Loss {
  double a, b, k2, h;
  explicit HuberLossWithCutoff(double a, double k) : a(a), b(a * a), k2(k * k), h(2.0 * a * k - b) {
    BASPACHO_CHECK_GT(a, 0.0);
    BASPACHO_CHECK_GE(k, a);
  }

  void setSize(double _a, double _k) {
    BASPACHO_CHECK_GT(_a, 0.0);
    BASPACHO_CHECK_GE(_k, _a);
    a = _a;
    b = _a * _a;
    k2 = _k * _k;
    h = 2.0 * a * _k - b;
  }

  inline double val(double s) const {
    if (s > k2) {
      return h;
    } else if (s > b) {
      const double r = std::sqrt(s);
      return 2.0 * a * r - b;
    } else {
      return s;
    }
  }

  inline double der(double s) const {
    if (s > k2) {
      return 0.0;
    } else if (s > b) {
      const double r = std::sqrt(s);
      return a / r;
    } else {
      return 1.0;
    }
  }

  inline std::pair<double, double> jet2(double s) const {
    if (s > k2) {
      return {h, 0.0};
    } else if (s > b) {
      const double r = std::sqrt(s);
      const double d = a / r;
      return {2.0 * a * r - b, d};
    } else {
      return {s, 1.0};
    }
  }

  inline std::tuple<double, double, double> jet3(double s) const {
    if (s > k2) {
      return {h, 0.0, 0.0};
    } else if (s > b) {
      const double r = std::sqrt(s);
      const double d = a / r;
      return {2.0 * a * r - b, d, -d / (2.0 * s)};
    } else {
      return {s, 1.0, 0.0};
    }
  }
};

struct CauchyLoss : Loss {
  double b, c;
  explicit CauchyLoss(double a) : b(a * a), c(1 / b) {}

  void setSize(double a) {
    b = a * a;
    c = 1 / b;
  }

  inline double val(double s) const {
    double sum = 1.0 + s * c;
    return b * std::log(sum);
  }

  inline double der(double s) const {
    double sum = 1.0 + s * c;
    return 1.0 / sum;
  }

  inline std::pair<double, double> jet2(double s) const {
    double sum = 1.0 + s * c;
    double inv = 1.0 / sum;
    return {b * std::log(sum), inv};
  }

  inline std::tuple<double, double, double> jet3(double s) const {
    double sum = 1.0 + s * c;
    double inv = 1.0 / sum;
    return {b * std::log(sum), inv, -c * (inv * inv)};
  }
};

} // namespace small_thing
