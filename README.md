# force_control
Implementations of 6D Cartesian space admittance control. Supports hybrid force-velocity control.

The algorithm creates a virtual spring-mass-damper system using a position controlled robot. You can specify the following parameters:
* 6x6 stiffness matrix, inertia matrix, damper matrix.
You can update the following online:
* Direction and dimension of force (soft) and position (rigid) control axes

Hardware requirements:
* A robot arm with high accuracy (high stiffness) position control interface.
* A wrist mounted FT sensor.

Author: Yifan Hou
yifanhou at stanford dot edu

# Install
## Dependency
Please install the following packages:
* [cpplibrary](https://github.com/yifan-hou/cpplibrary)

## Build
``` sh
cd force_control
mkdir build && cd build
cmake ..
make -j
make install
```

# How to use
## Use with cmake
``` makefile
# replace ${CMAKE_INSTALL_PREFIX} with your install location
find_library(FORCE_CONTROLLERS FORCE_CONTROLLERS HINTS ${CMAKE_INSTALL_PREFIX}/lib/)
find_library(RUT Utilities HINTS ${CMAKE_INSTALL_PREFIX}/lib/RobotUtilities)

# your executable
add_executable(force_control_demo src/main.cc)
target_link_libraries(force_control_demo
  ${RUT}
  ${FORCE_CONTROLLERS}
)
```

## c++ code sketch
Headers:
``` c++
#include <RobotUtilities/spatial_utilities.h>
#include <force_control/admittance_controller.h>

typedef Eigen::Matrix<double, 6, 1> Vector6d;
```
Create the controller config, initialize controller:
``` c++
AdmittanceController::AdmittanceControllerConfig admittance_config;
// populate admittance_config before using it

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