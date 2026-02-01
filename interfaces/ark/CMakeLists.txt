#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

add_library(ark_sensor_model
   camera_model/CameraModelParam.cpp
   camera_model/CameraModelParam.h
   camera_model/RandomWalkCov.cpp
   camera_model/RandomWalkCov.h
   extrinsics_model/RandomWalkCov.cpp
   extrinsics_model/RandomWalkCov.h
   imu_model/ImuCalibParam.cpp
   imu_model/ImuCalibParam.h
   imu_model/RandomWalkCov.cpp
   imu_model/RandomWalkCov.h
   preint/PreIntegration.h
   rolling_shutters/RollingShutterData.h
)
set_property(TARGET ark_sensor_model PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ark_sensor_model mps device_calibration viba_common imu_types preintegration small_thing)
target_include_directories(ark_sensor_model PUBLIC ".")


add_library(point_observation
   point_observation/PointObservation.h
   point_observation/PointObservationFormat.h
   point_observation/PointObservationReader.cpp
   point_observation/PointObservationReader.h
   point_observation/PointObservationWriter.cpp
   point_observation/PointObservationWriter.h
)
add_dependencies(point_observation fast-cpp-csv-parser)
set_property(TARGET point_observation PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(point_observation PUBLIC ".")


add_library(ark_session_data
    session_data/ImuCalibConversion.cpp
    session_data/SessionData.cpp
    session_data/ImuCalibConversion.h
    session_data/SessionData.h
)
set_property(TARGET ark_session_data PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ark_session_data ark_sensor_model point_observation)
target_include_directories(ark_session_data PUBLIC ".")

add_library(ark_io
   io/SaveDeviceTrajectory.cpp
   io/SaveDeviceTrajectory.h
   io/SaveOnlineCalib.cpp
   io/SaveOnlineCalib.h
)
set_property(TARGET ark_io PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ark_io ark_single_session)
target_include_directories(ark_io PUBLIC ".")


add_executable(ark_vi_ba main_AriaKit_ViBa.cpp)
target_link_libraries (ark_vi_ba ark_io CLI11::CLI11)

add_executable(process_vrs main_ProcessVRS.cpp)
target_link_libraries (process_vrs vrs_data_provider vrs_utils mps imu_types CLI11::CLI11)

# Check for Graphic context availability
find_package(OpenGL)
if(UNIX AND NOT APPLE)
    # On Linux, we nees to ensure X11 is available for OpenGL/GLX
    find_package(X11)
endif()

if(${OpenGL_FOUND} AND (NOT UNIX OR APPLE OR X11_FOUND)) 
    set(COMPILE_GUI_APP True)
else()
    set(COMPILE_GUI_APP False)
endif()

if(COMPILE_GUI_APP)
    message(STATUS "Everything for building gui library was found.")
    # ImGui + Sokol + ImPlot  library
    add_library(imgui_sokol
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${implot_SOURCE_DIR}/implot.cpp
        ${implot_SOURCE_DIR}/implot_items.cpp
        gui/sokol_app_imgui.cpp
    )

    target_include_directories(imgui_sokol PUBLIC
        ${sokol_SOURCE_DIR}
        ${imgui_SOURCE_DIR}
        ${implot_SOURCE_DIR}
    )

    target_compile_definitions(imgui_sokol PUBLIC "ImDrawIdx=unsigned int")
    target_link_libraries(imgui_sokol PUBLIC OpenGL::GL)
    if (APPLE)
        target_link_libraries(imgui_sokol PUBLIC
            "-framework Cocoa"
            "-framework QuartzCore"
            "-framework OpenGL"
        )
        set_source_files_properties(gui/sokol_app_imgui.cpp PROPERTIES
            COMPILE_FLAGS "-x objective-c++"
        )
    elseif(UNIX AND NOT APPLE)
        target_link_libraries(imgui_sokol PUBLIC X11::X11 X11::Xcursor X11::Xi ${OpenGL_GLX_LIBRARY})
    endif()

    set_property(TARGET imgui_sokol PROPERTY POSITION_INDEPENDENT_CODE ON)
    add_executable(ark_vi_ba_gui
        main_AriaKit_ViBa_GUI.cpp
        gui/Viewer3D.cpp
        gui/Viewer3D.h
        gui/Geometry.cpp
        gui/Geometry.h
        gui/Camera.h
        gui/Camera.cpp
    )
    target_link_libraries(ark_vi_ba_gui PRIVATE imgui_sokol ark_io CLI11::CLI11 OpenGL::GL)
    target_include_directories(ark_vi_ba_gui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    # Enable C++17 for the GUI
    set_property(TARGET ark_vi_ba_gui PROPERTY CXX_STANDARD 17)

    # On macOS, compile as Objective-C++ for Cocoa support

else()
    message(WARNING "OpenGL not found - skipping GUI targets (imgui_sokol, ark_vi_ba_gui)")
endif()
