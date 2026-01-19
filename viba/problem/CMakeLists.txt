# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# NOTE: we define one target per interface, not an issue at this time

# problem, built with ark interface
add_library(ark_problem
    BaseMapVisualFactor.h
    Histograms.h
    MultiSessionHistogramImpl.h
    MultiSessionProblem.h
    MultiSessionProblemImpl.h
    PointRefinement.h
    SingleSessionProblem.h
    Stats.h
    Types.h
    BaseMapVisualFactor.cpp
    Histograms.cpp
    InertialFactor.cpp
    OmegaPriorFactor.cpp
    PointRefinement.cpp
    PriorFactor.cpp
    RandomWalkFactor.cpp
    SingleSessionProblem.cpp
    Stats.cpp
    VisualFactor.cpp
)
set_property(TARGET ark_problem PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ark_problem ark_sensor_model imu_types small_thing viba_common)
target_include_directories(ark_problem PUBLIC "../..")
