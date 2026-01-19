/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <session_data/SessionData.h>
#include <viba/problem/SingleSessionProblem.h>
#include <filesystem>

namespace visual_inertial_ba {

void saveOnlineCalib(
    const SingleSessionProblem& prob,
    const SessionData& fData,
    const std::filesystem::path& outputPath);

} // namespace visual_inertial_ba
