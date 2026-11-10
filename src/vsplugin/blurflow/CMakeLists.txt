cmake_minimum_required(VERSION 3.20)

project(blurflow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# a vapoursynth plugin. opencl and nvidia's decoder are loaded at runtime, so there is nothing to
# link against
add_library(blurflow MODULE blurflow.cpp chroma.cpp source.cpp)
set_target_properties(blurflow PROPERTIES PREFIX "")

if(NOT WIN32)
  target_link_libraries(blurflow PRIVATE ${CMAKE_DL_LIBS})
endif()
