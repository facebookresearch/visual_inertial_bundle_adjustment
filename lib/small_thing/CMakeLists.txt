# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

add_library(small_thing
  PCG.cpp
  Optimizer.cpp
  Print.cpp
  AtomicOps.h           
  CondensedFactor.h     
  Optimizer.h             
  Preconditioner.h        
  Proxies.h               
  Variable.h
  Common.h                
  Factor.h                
  PCG.h                   
  Print.h                 
  SoftLoss.h)
set_property(TARGET small_thing PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(small_thing ${BASPACHO_LIBRARY})
target_include_directories(small_thing PUBLIC "../")

add_subdirectory("tests")
