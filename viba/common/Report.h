/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <small_thing/Optimizer.h>
#include <filesystem>

namespace visual_inertial_ba {

void writeJsonReport(
    const std::filesystem::path& outputFilePath,
    const small_thing::Optimizer::Summary& summary);

} // namespace visual_inertial_ba
