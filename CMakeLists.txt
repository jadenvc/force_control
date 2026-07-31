cmake_minimum_required(VERSION 3.20)

project(
  force_control
  VERSION 0.1.0
  DESCRIPTION "6D Cartesian admittance and hybrid force-velocity control"
  LANGUAGES CXX
)

include(CMakePackageConfigHelpers)
include(CTest)
include(FetchContent)
include(GNUInstallDirs)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# These archives match the dependencies used by the audited development
# environment. SHA-256 verification makes any upstream archive change fail
# loudly instead of silently changing the build.
FetchContent_Declare(
  force_control_eigen_source
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
  SOURCE_SUBDIR _force_control_source_only
)
FetchContent_MakeAvailable(force_control_eigen_source)

set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_FORMAT_SOURCE OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL ON CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  force_control_yaml_cpp
  URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz
  URL_HASH SHA256=fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16
)
FetchContent_MakeAvailable(force_control_yaml_cpp)

FetchContent_Declare(
  force_control_robot_utilities_source
  URL https://github.com/yifan-hou/cpplibrary/archive/e01dc4ccd68363a571e1f6a3c8cd3ba6dbec130c.tar.gz
  URL_HASH SHA256=a224b3387ba1348d83433663e1fd6661791e413e1a1470a48b5eb7ac09771849
  SOURCE_SUBDIR _force_control_source_only
)
FetchContent_MakeAvailable(force_control_robot_utilities_source)

add_library(force_control_eigen INTERFACE)
add_library(force_control::Eigen ALIAS force_control_eigen)
set_target_properties(force_control_eigen PROPERTIES EXPORT_NAME Eigen)
target_include_directories(
  force_control_eigen
  INTERFACE
    "$<BUILD_INTERFACE:${force_control_eigen_source_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/eigen3>"
)

# force_control only uses the spatial and timing portions of cpplibrary. Keeping
# them in their own target preserves the existing libUtilities API while
# avoiding unrelated platform-specific utilities.
add_library(
  RobotUtilities SHARED
  "${force_control_robot_utilities_source_SOURCE_DIR}/src/spatial_utilities.cpp"
  "${force_control_robot_utilities_source_SOURCE_DIR}/src/timer_linux.cpp"
)
add_library(force_control::RobotUtilities ALIAS RobotUtilities)
set_target_properties(
  RobotUtilities
  PROPERTIES
    EXPORT_NAME RobotUtilities
    OUTPUT_NAME Utilities
)
target_compile_features(RobotUtilities PUBLIC cxx_std_17)
target_include_directories(
  RobotUtilities
  PUBLIC
    "$<BUILD_INTERFACE:${force_control_robot_utilities_source_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(
  RobotUtilities
  PUBLIC
    force_control_eigen
    yaml-cpp::yaml-cpp
)

add_library(
  FORCE_CONTROLLERS SHARED
  src/admittance_controller.cpp
  src/config_deserialize.cpp
)
add_library(force_control::force_control ALIAS FORCE_CONTROLLERS)
set_target_properties(
  FORCE_CONTROLLERS
  PROPERTIES
    EXPORT_NAME force_control
    OUTPUT_NAME FORCE_CONTROLLERS
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH "$ORIGIN"
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
)
target_compile_features(FORCE_CONTROLLERS PUBLIC cxx_std_17)
target_include_directories(
  FORCE_CONTROLLERS
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(
  FORCE_CONTROLLERS
  PUBLIC
    RobotUtilities
    force_control_eigen
    yaml-cpp::yaml-cpp
)

if(BUILD_TESTING)
  add_executable(force_control_smoke_test tests/smoke_test.cpp)
  target_link_libraries(force_control_smoke_test PRIVATE force_control::force_control)
  target_compile_definitions(
    force_control_smoke_test
    PRIVATE
      FORCE_CONTROL_EXAMPLE_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/example.yaml"
  )
  add_test(NAME force_control.smoke COMMAND force_control_smoke_test)
endif()

install(
  TARGETS FORCE_CONTROLLERS RobotUtilities force_control_eigen
  EXPORT force_controlTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(
  FILES
    "${force_control_robot_utilities_source_SOURCE_DIR}/include/RobotUtilities/spatial_utilities.h"
    "${force_control_robot_utilities_source_SOURCE_DIR}/include/RobotUtilities/timer_linux.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/RobotUtilities"
)
install(
  DIRECTORY "${force_control_eigen_source_SOURCE_DIR}/Eigen"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/eigen3"
)
install(
  FILES LICENSE
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/force_control/licenses"
  RENAME force_control-LICENSE
)
install(
  FILES "${force_control_robot_utilities_source_SOURCE_DIR}/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/force_control/licenses"
  RENAME RobotUtilities-LICENSE
)
install(
  FILES "${force_control_yaml_cpp_SOURCE_DIR}/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/force_control/licenses"
  RENAME yaml-cpp-LICENSE
)
install(
  FILES
    THIRD_PARTY_NOTICES.md
    "${force_control_eigen_source_SOURCE_DIR}/COPYING.APACHE"
    "${force_control_eigen_source_SOURCE_DIR}/COPYING.BSD"
    "${force_control_eigen_source_SOURCE_DIR}/COPYING.GPL"
    "${force_control_eigen_source_SOURCE_DIR}/COPYING.LGPL"
    "${force_control_eigen_source_SOURCE_DIR}/COPYING.MINPACK"
    "${force_control_eigen_source_SOURCE_DIR}/COPYING.MPL2"
    "${force_control_eigen_source_SOURCE_DIR}/COPYING.README"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/force_control/licenses"
)
install(
  EXPORT force_controlTargets
  FILE force_controlTargets.cmake
  NAMESPACE force_control::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/force_control"
)

configure_package_config_file(
  cmake/force_controlConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/force_controlConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/force_control"
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/force_controlConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/force_controlConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/force_controlConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/force_control"
)
