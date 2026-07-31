#include <cmath>
#include <iostream>

#include <RobotUtilities/spatial_utilities.h>
#include <RobotUtilities/timer_linux.h>
#include <yaml-cpp/yaml.h>

#include "force_control/admittance_controller.h"
#include "force_control/config_deserialize.h"

namespace {

bool is_finite(const RUT::Vector7d& value) {
  return value.array().isFinite().all();
}

}  // namespace

int main() {
  YAML::Node document;
  try {
    document = YAML::LoadFile(FORCE_CONTROL_EXAMPLE_CONFIG_PATH);
  } catch (const std::exception& error) {
    std::cerr << "Unable to load example config: " << error.what() << '\n';
    return 1;
  }

  AdmittanceController::AdmittanceControllerConfig config;
  if (!deserialize(document["admittance_controller"], config)) {
    std::cerr << "Example config did not deserialize\n";
    return 1;
  }

  if (std::abs(config.dt - 0.002) > 1e-12 ||
      std::abs(config.compliance6d.stiffness(0, 0) - 100.0) > 1e-12 ||
      std::abs(config.compliance6d.inertia(5, 5) - 0.005) > 1e-12) {
    std::cerr << "Example config values changed unexpectedly\n";
    return 1;
  }

  RUT::Vector7d pose;
  pose << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0;

  RUT::Timer timer;
  AdmittanceController controller;
  if (!controller.init(timer.tic(), config, pose)) {
    std::cerr << "Controller initialization failed\n";
    return 1;
  }

  controller.setForceControlledAxis(RUT::Matrix6d::Identity(), 6);
  controller.setRobotStatus(pose, RUT::Vector6d::Zero());
  controller.setRobotReference(pose, RUT::Vector6d::Zero());

  RUT::Vector7d pose_command;
  if (!controller.step(pose_command) || !is_finite(pose_command)) {
    std::cerr << "Controller smoke step failed\n";
    return 1;
  }

  return 0;
}
