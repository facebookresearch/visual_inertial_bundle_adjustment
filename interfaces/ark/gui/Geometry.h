/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "sokol_gfx.h"

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace viba_gui {
namespace geometry {

struct GeometryBuffer {
  sg_buffer buffer{0};
  size_t count = 0;
};

// Buffer management
GeometryBuffer createBuffer(const std::vector<Eigen::Vector3f>& vertices);
void updateBuffer(GeometryBuffer& buf, const std::vector<Eigen::Vector3f>& vertices);
void destroyBuffer(GeometryBuffer& buf);

// Drawing functions (pipelines lazily initialized on first use)
void drawPoints(
    const GeometryBuffer& buf,
    const Eigen::Matrix4f& mvp,
    const Eigen::Vector3f& color = Eigen::Vector3f(0.3f, 0.7f, 0.3f),
    float alpha = 0.6f);

void drawLineStrip(
    const GeometryBuffer& buf,
    const Eigen::Matrix4f& mvp,
    const Eigen::Vector3f& color = Eigen::Vector3f(1.0f, 0.5f, 0.0f));

// Cleanup (optional - called at shutdown)
void shutdown();

} // namespace geometry
} // namespace viba_gui
