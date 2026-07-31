# force_control
Implementations of 6D Cartesian space admittance control. Supports hybrid force-velocity control.

The algorithm creates a virtual spring-mass-damper-friction system using a position controlled robot. You can specify the following parameters:
* 6x6 stiffness matrix, inertia matrix, damper matrix, friction vector.

You can update the following online:
* Direction and dimension of force (soft) and position (rigid) control axes

Hardware requirements:
* A robot arm with high accuracy (high stiffness) position control interface.
* A wrist mounted FT sensor.

Author: Yifan Hou
yifanhou at stanford dot edu
# What is admittance control?
Admittance control is one of the two ways (the other is impedance control) to implement compliance control. Compliance control refers to methods to make a robot act compliantly via feedback control. Compliance is usually described by a spring-mass-damper system. For more details on how is compliance achieved, the difference between admittance and impedance, please checkout my [lecture notes on Compliance Control](https://www.dropbox.com/scl/fi/4xg3notqen0wrbkyk59i1/Intro_to_compliance_control.pdf?rlkey=qrm58807j5q4irl2viyrp2df7&e=2&dl=0).

# What is hybrid force-velocity control?
A robot usually lives in a multi-dimensional space. For example, a typical robot arm moves its hand in 6D rigid body space. When doing compliance control, you need to specify the compliant behavior you want in all those six dimensions.

Hybrid force-velocity control (or hybrid force-position control, just different names) refers to the act of using different compliance parameters (inertia, stiffness, damping) in different directions in this 6D space. This package lets you specify parameters in all six dimensions, as well as the directions of the six axes.

If you don't want HFVC and just want uniform compliance, that is easy to set too. See the examples below for how to do it.

# SAFETY WARNING
Force control is a high rate, high order control scheme that can go very wrong very quickly. Make sure you understand what you are doing before using this code. If the compliance parameters are not suitable, e.g. the robot is configured to be too soft and light while force feedback is not well calibrated, the robot will drift away very fast, which can be dangerous. 

For your own safety, the following steps are recommended before launching a force-controlled robot:
1. Start from enabling only one translational compliance axis (using `setForceControlledAxis`). Get the compliance control to work, get a feeling of what parameters make sense for your robot before enabling more axes. Common mistakes to pay attention to:
  * Force feedback transformation is wrong. This could cause a positive feedback loop.
  * Force sensor is badly calibrated.
  * Compliance parameters are set to be too senstitive (unstable motion) or too insensitive (no response to external force).
2. Safe parameters to use when testing for the first time:
  * Set the inertia value to the same magnitude as the actual robot mass. For example, 2~5kg is reasonable for a typical table top robot arm like ABB120, UR5e.
  * Set damping to a small value, e.g. 0.1
  * Set stiffness to a reasonable value.
  * direct_force_control_gains and direct_force_control_I_limit should be all zero.
3. Make sure the robot stays clear from any potential collisions.
4. Make sure you have the emergency stop button at your thumb.
5. Start to run the controller.
  * Stop immediately if there is any sudden/unstable motion.
  * If the robot appears to be stable, gently push the robot in the direction where you enabled compliance. Check if the robot can be dragged as expected. If not, check your sensor sign/transformation/robot tool frame setting, etc.
  * If the signs/direction seems fine but the robot is just shaking a bit, graduately increase damping.
6. Now you have a one-axis compliance control working. You can play with the parameters as you wish, e.g. graduately reduce the inertia values and damping to get a more "soft" feeling.
7. Enable all three translational axes.
8. Redo the above for rotational axes. Note the order of magnitude of parameters are quiet different between rotational and translational axes.

# Install

## Reproducible dependencies

The default CMake build downloads and verifies the same core dependency
versions used by the audited development environment:

* Eigen 3.4.0
* yaml-cpp 0.8.0
* [cpplibrary](https://github.com/yifan-hou/cpplibrary) commit
  `e01dc4ccd68363a571e1f6a3c8cd3ba6dbec130c`

Each source archive is protected by a SHA-256 checksum in
[`CMakeLists.txt`](CMakeLists.txt). Nothing needs to be installed under
`/usr/local` or `~/.local` before configuring this project.

The baseline system used for the container and CI build is Ubuntu 22.04 with
GCC 11 and C++17. The Dockerfile pins the multi-platform Ubuntu image by digest.

## Native build

Install CMake 3.20 or newer, a C++ compiler, and Git. On Ubuntu 22.04:

```sh
sudo apt-get update
sudo apt-get install --yes build-essential ca-certificates cmake git
```

Then configure, build, test, and install:

```sh
git clone https://github.com/jadenvc/force_control.git
cd force_control
cmake --preset release
cmake --build --preset release
ctest --preset release
cmake --install build/release
```

The preset installs into `build/install`, keeping the host system unchanged.
Use `--prefix /your/prefix` with the final command to choose another
installation location.

## Container build

The checked-in `Dockerfile` performs the same release build and smoke test:

```sh
docker build --tag force-control:0.1.0 .
```

## FlipUp MuJoCo environment

The standalone UR5e + WSG50 book-pivot environment and all required MuJoCo
assets are included in [`flipup_minimal`](flipup_minimal). The optional
Force Dimension haptic bridge is in [`teleop`](teleop).

Use a current Python 3.10-or-newer environment rather than trying to reproduce
one workstation package-for-package. On Ubuntu, a minimal setup is:

```sh
sudo apt-get install python3-venv libgl1 libglfw3 libosmesa6 ffmpeg
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --editable ./flipup_minimal
python -m pip install --requirement teleop/requirements.txt
```

Run the environment without hardware or a viewer:

```sh
python teleop/teleop_flipup.py --dry-run --no-view
```

For physical haptics, install a compatible Force Dimension SDK separately,
install [`teleop/50-forcedimension.rules`](teleop/50-forcedimension.rules), and
set `FORCEDIMENSION_LIB` to the SDK's `libdrd.so.3` if it is not on the system
library path. See [`teleop/README.md`](teleop/README.md) for details.

## Optional plotting environment

The controller library itself does not require Python. To reproduce the
environment used by `scripts/plot.py`, create a virtual environment and install
the two pinned plotting packages:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --requirement requirements-plot.txt
```

# How to use

## Use with cmake

After installing, point `CMAKE_PREFIX_PATH` at the chosen prefix and consume the
exported target:

```cmake
find_package(force_control 0.1 CONFIG REQUIRED)

add_executable(force_control_demo src/main.cc)
target_link_libraries(force_control_demo PRIVATE force_control::force_control)
```

## config example

A complete, smoke-tested configuration is checked in at
[`config/example.yaml`](config/example.yaml):

``` yaml
admittance_controller:
  dt: 0.002
  log_to_file: false
  log_file_path: "/tmp/admittance_controller.log"
  alert_overrun: false
  compliance6d:
    stiffness: [100, 100, 100, 1, 1, 1]
    damping: [2, 2, 2, 0.2, 0.2, 0.2]
    inertia: [5, 5, 5, 0.005, 0.005, 0.005]
    stiction: [0, 0, 0, 0, 0, 0]
  max_spring_force_magnitude: 50
  max_spring_torque_magnitude: 4
  direct_force_control_gains:
    P_trans: 0
    I_trans: 0
    D_trans: 0
    P_rot: 0
    I_rot: 0
    D_rot: 0
  direct_force_control_I_limit: [0, 0, 0, 0, 0, 0]
```

MuJoCo is not a dependency of the C++ force-control library. The bundled
FlipUp environment is an independent Python package and does not change the
native library build.

## c++ code example
Headers:
``` c++
#include <RobotUtilities/spatial_utilities.h>
#include <force_control/admittance_controller.h>

typedef Eigen::Matrix<double, 6, 1> Vector6d;
```
Create the controller config, initialize controller:
``` c++
// load config
AdmittanceController::AdmittanceControllerConfig admittance_config;
const std::string CONFIG_PATH = "path_to/config.yaml";
YAML::Node config{};
try {
  config = YAML::LoadFile(CONFIG_PATH);
  deserialize(config["admittance_controller"], admittance_config);
} catch (const std::exception& e) {
  std::cerr << "Failed to load the config file: " << e.what() << std::endl;
  return -1;
}

AdmittanceController controller;

RUT::Timer timer;
RUT::TimePoint time0 = timer.tic();
RUT::Vector7d pose, pose_ref, pose_cmd;
RUT::Vector6d wrench, wrench_WTr;

controller.init(time0, admittance_config, pose);
```
Set force control axes and dimension. There are a couple options:
``` c++
// Regular admittance control, all 6 axes are force dimensions:
RUT::Matrix6d Tr = RUT::Matrix6d::Identity();
int n_af = 6;
controller.setForceControlledAxis(Tr, n_af);

// HFVC, compliant translational motion, rigid rotation motion 
RUT::Matrix6d Tr = RUT::Matrix6d::Identity();
int n_af = 3;
controller.setForceControlledAxis(Tr, n_af);

// HFVC, compliant rotational motion, rigid translational motion
RUT::Matrix6d Tr;
Tr << 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0;
int n_af = 3;
controller.setForceControlledAxis(Tr, n_af);

// n_af = 0 disables compliance. All axes uses rigid motion
RUT::Matrix6d Tr = RUT::Matrix6d::Identity();
int n_af = 0;
controller.setForceControlledAxis(Tr, n_af);
```
Now we are ready to start the control loop. Assuming we have access to a `robot_ptr` object that can provides pose and wrench feedback.

``` c++
pose_ref = pose;
wrench_WTr.setZero();

timer.tic();

while (true) {
    // Update robot status
    robot_ptr->getCartesian(pose);
    robot_ptr->getWrenchTool(wrench);
    controller.setRobotStatus(pose, wrench);

    // Update robot reference
    controller.setRobotReference(pose_ref, wrench_WTr);

    // Compute the control output
    controller.step(pose_cmd);

    // send action to robot
    robot_ptr->setCartesian(pose_cmd);
    
    // sleep till next iteration
    spin();
}
```

## Reference
This package was implemented as a part of 
Y. Hou and M. T. Mason, "Robust Execution of Contact-Rich Motion Plans by Hybrid Force-Velocity Control,"
2019 International Conference on Robotics and Automation (ICRA), Montreal, QC, Canada, 2019, pp. 1933-1939

The implementation was initially based on James A. Maples and Joseph J. Becker, "Experience in Force Control of Robotic Manipulators", 
Then a lot more functionalities were added. Please contact yifan for questions.
