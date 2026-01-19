/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <small_thing/Optimizer.h>

namespace visual_inertial_ba {

// run point refinement (using factor introspection)
void refinePoints(small_thing::Optimizer& opt, bool muted = false);

} // namespace visual_inertial_ba
