# Reproducing the audited environment

This repository reproduces the standalone `force_control` C++ library and
includes the self-contained FlipUp MuJoCo environment. It does not copy build
products, Conda environments, robot calibration, experiment datasets, or logs
from the development machine.

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

## FlipUp environment

The C++ library remains independent of MuJoCo. The Python additions are:

| Concern | Location |
| --- | --- |
| MuJoCo scene, controller, heuristic, meshes, and textures | `flipup_minimal/` |
| Force Dimension haptic bridge and device-free dry run | `teleop/` |
| Optional Zarr dataset support | `teleop/requirements_dataset.txt` |

The Python package declares broad compatible dependency versions so a new
machine can resolve packages appropriate for its platform. Python 3.10,
MuJoCo 3.2.7, and dm-control 1.0.27 are a known-good reference, not a mandatory
byte-for-byte lock. Follow the commands in `README.md` and validate with seed 0.

The proprietary Force Dimension SDK, robot calibration, experiment datasets,
and generated videos remain external. Large logs and hardware-specific state
must not be committed.
