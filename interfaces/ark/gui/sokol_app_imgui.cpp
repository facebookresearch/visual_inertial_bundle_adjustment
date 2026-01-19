/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// this is the file that compiles the implementation of the sokol library

#define SOKOL_IMPL
#define SOKOL_TRACE_HOOKS
#define SOKOL_GLCORE

// core sokol lib (order independent)
#include "sokol_app.h"
#include "sokol_args.h"
#include "sokol_fetch.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"
#include "sokol_log.h"

#include "imgui.h"

// util libraries
#include "util/sokol_imgui.h"

#include "util/sokol_gfx_imgui.h"
#include "util/sokol_gl.h"
