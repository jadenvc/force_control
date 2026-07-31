#include <force_control/admittance_controller.h>

int main() {
  AdmittanceController controller;
  AdmittanceController::AdmittanceControllerConfig config;
  return config.dt > 0.0 ? 0 : 1;
}
