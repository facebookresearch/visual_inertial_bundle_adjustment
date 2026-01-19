/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <small_thing/Print.h>

#include <array>
#include <cmath>
#include <iomanip>

namespace small_thing {

using namespace std;

std::string indent(const std::string& text) {
  const int indentLevel = 4;
  const std::string indent(indentLevel, ' ');

  std::stringstream ss(text);
  std::string line;
  std::stringstream ress;
  bool firstLine = true;

  while (std::getline(ss, line)) {
    if (!firstLine) {
      ress << '\n';
    }
    if (!line.empty()) {
      ress << indent << line;
    }
    firstLine = false;
  }

  return ress.str();
}

string percentageString(double rat, int precision) {
  stringstream ss;
  ss << fixed << setprecision(precision) << (rat * 100) << "%";
  return ss.str();
}

string humanReadableSize(size_t nbytes) {
  stringstream ss;
  static const array<string, 5> suffixes = {"b", "Kb", "Mb", "Gb", "Tb"};
  double num = nbytes;
  unsigned int i = 0;
  while ((num >= 256) && (i < suffixes.size() - 1)) {
    i += 1;
    num /= 1024.0;
  }
  ss << fixed << setprecision(i == 0 ? 0 : (num < 1 ? 2 : 1)) << num << suffixes[i];
  return ss.str();
}

string microsecondsString(size_t microseconds, int precision) {
  ostringstream os;
  constexpr size_t kMilliSecondUs = 1000;
  constexpr size_t kTenthOfASecondUs = 100 * kMilliSecondUs;
  constexpr size_t kSecondUs = 1000 * kMilliSecondUs;
  constexpr size_t kMinuteUs = 60 * kSecondUs;
  constexpr size_t kHourUs = 60 * kMinuteUs;
  if (microseconds < kMilliSecondUs) {
    os << microseconds << "\u03bcs";
  } else if (microseconds < kTenthOfASecondUs) {
    os << fixed << setprecision(precision) << ((double)microseconds / kMilliSecondUs) << "ms";
  } else {
    if (microseconds >= kHourUs) {
      os << microseconds / kHourUs << "h";
      microseconds %= kHourUs;
      if (microseconds >= kMinuteUs) {
        os << microseconds / kMinuteUs << "m";
      }
    } else if (microseconds >= kMinuteUs) {
      os << microseconds / kMinuteUs << "m";
      microseconds %= kMinuteUs;
      size_t seconds = (size_t)round((double)microseconds / kSecondUs);
      if (seconds > 0) {
        os << seconds << "s";
      }
    } else {
      os << fixed << setprecision(precision) << ((double)microseconds / kSecondUs) << "s";
    }
  }
  return os.str();
}

} // namespace small_thing
