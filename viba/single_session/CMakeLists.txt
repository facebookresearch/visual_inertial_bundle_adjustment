# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# NOTE: we define one target per interface, not an issue at this time

# single session, built with ark interface
add_library(ark_single_session
   EvalCalibration.cpp
   FactoryCalibPriors.cpp
   InertialFactors.cpp
   InitCalibration.cpp
   InitPointTracks.cpp
   InitRescaling.cpp
   InitRigs.cpp
   Matcher.cpp
   OmegaPriors.cpp
   RandomWalkFactors.cpp
   SingleSessionAdapter.cpp
   Triangulation.cpp
   VisualFactors.cpp
   Matcher.h
   SingleSessionAdapter.h
   TrajectoryBase.h
   Triangulation.h
)
set_property(TARGET ark_single_session PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ark_single_session ark_session_data ark_problem)
target_include_directories(ark_single_session PUBLIC "../..")
