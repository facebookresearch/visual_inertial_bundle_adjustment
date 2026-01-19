/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <baspacho/baspacho/Utils.h>

[[noreturn]] inline void throwError(const char* file, int line, const char* msg) {
  std::stringstream s;
  s << "[" << ::BaSpaCho::timeStamp() << " " << file << ":" << line << "] Check failed: " << msg;
  throw std::runtime_error(s.str());
}

// Same as above but returns a const reference.
template <class C>
typename C::value_type::second_type& findOrDieWithInfo(
    C& c,
    const typename C::value_type::first_type& key,
    const char* file,
    int line,
    const char* msg) {
  if (auto it = c.find(key); it != c.end()) {
    return it->second;
  }
  throwError(file, line, msg);
}

template <class C>
const typename C::value_type::second_type& findOrDieWithInfo(
    const C& c,
    const typename C::value_type::first_type& key,
    const char* file,
    int line,
    const char* msg) {
  if (auto it = c.find(key); it != c.end()) {
    return it->second;
  }
  throwError(file, line, msg);
}

template <class C>
typename C::value_type::second_type& findOrDieWithInfo2(
    C& c1,
    C& c2,
    const typename C::value_type::first_type& key,
    const char* file,
    int line,
    const char* msg) {
  if (auto it = c1.find(key); it != c1.end()) {
    return it->second;
  }
  if (auto it2 = c2.find(key); it2 != c2.end()) {
    return it2->second;
  }
  throwError(file, line, msg);
}

template <class C>
const typename C::value_type::second_type& findOrDieWithInfo2(
    const C& c1,
    const C& c2,
    const typename C::value_type::first_type& key,
    const char* file,
    int line,
    const char* msg) {
  if (auto it = c1.find(key); it != c1.end()) {
    return it->second;
  }
  if (auto it2 = c2.find(key); it2 != c2.end()) {
    return it2->second;
  }
  throwError(file, line, msg);
}

// NOTE: use varargs to handle the case of multiple commas confusing
// the preprocessor such as in: findOrDie(container, {k1, k2})
#define findOrDie(...) \
  findOrDieWithInfo(__VA_ARGS__, __FILE__, __LINE__, "findOrDie(" #__VA_ARGS__ ")")

#define findOrDie2(...) \
  findOrDieWithInfo2(__VA_ARGS__, __FILE__, __LINE__, "findOrDie2(" #__VA_ARGS__ ")")
