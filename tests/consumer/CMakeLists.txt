cmake_minimum_required(VERSION 3.20)

project(force_control_consumer LANGUAGES CXX)

find_package(force_control 0.1 CONFIG REQUIRED)

add_executable(force_control_consumer main.cpp)
target_link_libraries(
  force_control_consumer
  PRIVATE force_control::force_control
)
