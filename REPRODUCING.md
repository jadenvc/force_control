# Reproducing the audited environment

This repository intentionally reproduces the standalone `force_control` C++
library. It does not copy build products, Conda environments, robot
calibration, MuJoCo assets, or logs from the development machine.

## Audited baseline

The reproducibility work started from `mainline` commit
`460bc3bdc6036e7c531cbe7751692d05bb36052d` on Ubuntu 22.04.5 with GCC 11.4.
The default build captures the source inputs that affected the library:

| Input | Pin |
| --- | --- |
| C++ language level | C++17 |
| Eigen | 3.4.0 archive plus SHA-256 |
| yaml-cpp | 0.8.0 archive plus SHA-256 |
| RobotUtilities | cpplibrary commit `e01dc4ccd68363a571e1f6a3c8cd3ba6dbec130c` plus SHA-256 |
| Container OS | Ubuntu 22.04 multi-platform image digest in `Dockerfile` |
| Optional plotting | Exact versions in `requirements-plot.txt` |

The archive checksums and build graph are defined in `CMakeLists.txt`. Generated
files under `build/` are deliberately ignored.

## Verification

Run the complete native verification:

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
cmake --install build/release
cmake \
  -S tests/consumer \
  -B build/consumer \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/build/install"
cmake --build build/consumer
./build/consumer/force_control_consumer
```

Alternatively, `docker build --tag force-control:0.1.0 .` performs configure,
build, test, and install inside the pinned Ubuntu base image.

## Environment boundaries

MuJoCo was not linked, imported, or referenced by `force_control`. In the
audited workstation it belonged to a separate stack:

| Concern | Repository | Audited runtime |
| --- | --- | --- |
| MuJoCo environments and MJCF assets | [PyriteEnvSuites](https://github.com/yifan-hou/PyriteEnvSuites) | Python 3.10.20, MuJoCo 3.2.7, dm-control 1.0.27 |
| Real-robot adapters and deployed controller YAML | [hardware_interfaces](https://github.com/yifan-hou/hardware_interfaces) | Robot- and workstation-specific |
| Task type conversions | [PyriteConfig](https://github.com/yifan-hou/PyriteConfig) | Pyrite-specific |

Those projects should be reproduced through a separate umbrella repository
with commit-pinned submodules and their own environment lock. In particular,
large experiment logs and hardware-specific state should never be copied into
this repository.
