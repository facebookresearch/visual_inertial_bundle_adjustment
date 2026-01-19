/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iterator>
#include <tuple>

// enumeration convenience: `for (auto [i, v] : enumerate(vec)) { ... }`
template <
    typename T,
    typename TIter = decltype(std::begin(std::declval<T>())),
    typename = decltype(std::end(std::declval<T>()))>
constexpr auto enumerate(T&& iterable) {
  struct iterator {
    size_t i;
    TIter iter;
    bool operator!=(const iterator& other) const {
      return iter != other.iter;
    }
    void operator++() {
      ++i;
      ++iter;
    }
    auto operator*() const {
      return std::tie(i, *iter);
    }
  };
  struct iterable_wrapper {
    T iterable;
    auto begin() {
      return iterator{0, std::begin(iterable)};
    }
    auto end() {
      return iterator{0, std::end(iterable)};
    }
  };
  return iterable_wrapper{std::forward<T>(iterable)};
}
