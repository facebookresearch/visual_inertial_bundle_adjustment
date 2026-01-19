/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>

namespace visual_inertial_ba {
constexpr std::array<const char*, 9> PointObservationColumns = {
    "point_id",
    "capture_timestamp_ns",
    "camera_index",
    "projection_base_res_x",
    "projection_base_res_y",
    "sqrt_h_base_res_00",
    "sqrt_h_base_res_01",
    "sqrt_h_base_res_10",
    "sqrt_h_base_res_11",
};

} // namespace visual_inertial_ba
