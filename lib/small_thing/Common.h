/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <sophus/se3.hpp>
#include <Eigen/Geometry>
#include <memory>
#include <optional>
#include <typeindex>
#include <unordered_map>

namespace small_thing {

using LogFunc = std::function<void(const std::string&)>;

template <typename Tuple, typename AddArgs>
struct CompoundTuple;

template <typename... TupArgs, typename AddArgs>
struct CompoundTuple<std::tuple<TupArgs...>, AddArgs> {
  using type = std::tuple<TupArgs..., AddArgs>;
};

template <typename Tuple, typename AddArgs>
using CompoundTuple_t = typename CompoundTuple<Tuple, AddArgs>::type;

template <typename T>
struct UnpackOpt {
  static constexpr bool isOpt = false;
  using BaseType = T;
};

template <typename T>
struct UnpackOpt<std::optional<T>> {
  static constexpr bool isOpt = true;
  using BaseType = T;
};

template <typename T, typename F1, typename F2>
auto unpackingOptional(const T& t, F1&& f1, F2&& f2) {
  if constexpr (UnpackOpt<T>::isOpt) {
    return t.has_value() ? f1(*t) : f2();
  } else {
    return f1(t);
  }
}

template <typename T>
const typename UnpackOpt<T>::BaseType& unpackedOpt(const T& t) {
  if constexpr (UnpackOpt<T>::isOpt) {
    return *t;
  } else {
    return t;
  }
}

// A type used to encapsulate a compile-time int value. Ideally we would like to use templated
// lambdas, but they are C++20. In C++17 a lambda accepting a an `auto` argument will be fed a
// dummy argument which is an empty class of type IntWrap<n>, allowing us to recover n.
template <int i>
struct IntWrap {
  static constexpr int value = i;
};

// compile time loop, needed to iterate over tuples (which have different element types)
template <int i, int j, typename F>
void forEach(F&& f) {
  if constexpr (i < j) {
    f(IntWrap<i>());
    forEach<i + 1, j>(std::forward<F>(f));
  }
}

// call f with arguments of a static range (as wrapped ints)
template <int i, int j, typename F, typename... Args>
decltype(auto) withStaticRange(F&& f, Args&&... args) {
  if constexpr (i < j) {
    return withStaticRange<i + 1, j>(std::forward<F>(f), std::forward<Args>(args)..., IntWrap<i>());
  } else {
    return f(args...);
  }
}

// Storage allowing to store different types derived from Base, indexing on std::type_index
template <typename Base>
struct TypedStore {
  template <typename Derived>
  Derived& get() {
    static const std::type_index ti(typeid(Derived));
    std::unique_ptr<Base>& pStoreT = stores[ti];
    if (!pStoreT) {
      pStoreT.reset(new Derived);
    }
    return dynamic_cast<Derived&>(*pStoreT);
  }

  template <typename Derived>
  const Derived* getConst() const {
    static const std::type_index ti(typeid(Derived));
    auto it = stores.find(ti);
    if (it == stores.end()) {
      return nullptr;
    }
    return dynamic_cast<const Derived*>(it->second.get());
  }

  std::unordered_map<std::type_index, std::unique_ptr<Base>> stores;
};

// helper for unordered_map
struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

/// Get an Eigen reference pointing to a nullptr.
template <
    typename Scalar,
    int Rows = Eigen::Dynamic,
    int Columns = Eigen::Dynamic,
    int StorageOrder = Eigen::ColMajor>
constexpr Eigen::Ref<Eigen::Matrix<Scalar, Rows, Columns, StorageOrder>> nullRef(
    const int rows = Rows != Eigen::Dynamic ? Rows : 0,
    const int columns = Columns != Eigen::Dynamic ? Columns : 0) {
  return Eigen::Map<Eigen::Matrix<Scalar, Rows, Columns, StorageOrder>>(nullptr, rows, columns);
}

/// Test if an Eigen Reference is pointing to a nullptr.
template <typename Base>
constexpr bool isNull(const Eigen::Ref<Base>& ref) {
  return ref.data() == nullptr;
}

/**
 * A helper struct that allows direct assignments of NullRefs as function default
 * arguments.
 */
struct NullRef {
  /// The default constructor for general usecases.
  NullRef() = default;

  /// There might be edge cases where a dynamic sized NullRef is required. This
  /// provides the option, while it should generally not be used.
  NullRef(const int numRows, const int numCols) : numRows_(numRows), numCols_(numCols) {}

  /// Fixed size (intentionally implicit conversion):
  template <typename Scalar, int R, int C, int O, int MR, int MC>
  // @lint-ignore CLANGTIDY hicpp-explicit-conversions
  operator Eigen::Ref<Eigen::Matrix<Scalar, R, C, O, MR, MC>>() {
    return nullRef<Scalar, R, C, O>(R, C);
  }
  template <typename Scalar, int R, int C, int O, int MR, int MC>
  // @lint-ignore CLANGTIDY hicpp-explicit-conversions
  operator Eigen::Ref<const Eigen::Matrix<Scalar, R, C, O, MR, MC>>() {
    return nullRef<Scalar, R, C, O>(R, C);
  }

  /// Dynamic Matrices (intentionally implicit conversion):
  template <typename Scalar, int O, int MR, int MC>
  // @lint-ignore CLANGTIDY hicpp-explicit-conversions
  operator Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, O, MR, MC>>() {
    return nullRef<Scalar, Eigen::Dynamic, Eigen::Dynamic, O>(numRows_, numCols_);
  }
  template <typename Scalar, int O, int MR, int MC>
  // @lint-ignore CLANGTIDY hicpp-explicit-conversions
  operator Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, O, MR, MC>>() {
    return nullRef<Scalar, Eigen::Dynamic, Eigen::Dynamic, O>(numRows_, numCols_);
  }

  /// Fixed R + Dynamic Matrices (intentionally implicit conversion):
  template <typename Scalar, int R, int O, int MR, int MC>
  // @lint-ignore CLANGTIDY hicpp-explicit-conversions
  operator Eigen::Ref<Eigen::Matrix<Scalar, R, Eigen::Dynamic, O, MR, MC>>() {
    return nullRef<Scalar, R, Eigen::Dynamic, O>(R, numCols_);
  }
  template <typename Scalar, int R, int O, int MR, int MC>
  // @lint-ignore CLANGTIDY hicpp-explicit-conversions
  operator Eigen::Ref<const Eigen::Matrix<Scalar, R, Eigen::Dynamic, O, MR, MC>>() {
    return nullRef<Scalar, R, Eigen::Dynamic, O>(R, numCols_);
  }

  /// Fixed C + Dynamic Matrices (intentionally implicit conversion):
  template <typename Scalar, int C, int O, int MR, int MC>
  // @lint-ignore CLANGTIDY hicpp-explicit-conversions
  operator Eigen::Ref<Eigen::Matrix<Scalar, Eigen::Dynamic, C, O, MR, MC>>() {
    return nullRef<Scalar, Eigen::Dynamic, C, O>(numRows_, C);
  }
  template <typename Scalar, int C, int O, int MR, int MC>
  // @lint-ignore CLANGTIDY hicpp-explicit-conversions
  operator Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, C, O, MR, MC>>() {
    return nullRef<Scalar, Eigen::Dynamic, C, O>(numRows_, C);
  }

 private:
  const int numRows_ = 0;
  const int numCols_ = 0;
};

} // namespace small_thing
