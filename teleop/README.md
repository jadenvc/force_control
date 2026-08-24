# Haptic teleoperation

This directory adds Force Dimension omega teleoperation and force reflection to
the self-contained MuJoCo environments in `../flipup_minimal` and
`../conveyor_minimal`.

Per-task guides:

- [`README_flipup.md`](README_flipup.md) — the book pivot: controls, tuning,
  dataset collection, replay.
- [`README_conveyor.md`](README_conveyor.md) — the conveyor pick-place: the
  gripper channel, where this task's force actually is, and what has not yet
  been checked against hardware.

## Basic setup

Use Python 3.10 or newer. From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --editable ./flipup_minimal
python -m pip install --editable ./conveyor_minimal
python -m pip install --requirement teleop/requirements.txt
```

MuJoCo 3.2.7 and dm-control 1.0.27 are known-good reference versions, but the
package intentionally uses compatible version ranges so installation can adapt
to a newer machine.

Verify the complete control path without a haptic device:

```sh
python teleop/teleop_flipup.py --dry-run --no-view
```

```sh
python teleop/collect_conveyor.py --dry-run --episodes 1 --auto-finish
```

## Force Dimension hardware

The SDK and its shared libraries are proprietary prerequisites and are not
included. Install a compatible Force Dimension SDK, then either make
`libdrd.so.3` discoverable by the system loader or set:

```sh
export FORCEDIMENSION_LIB=/path/to/force-dimension-sdk/libdrd.so.3
```

On Linux, copy `50-forcedimension.rules` to `/etc/udev/rules.d/`, reload the
udev rules, reconnect the device, and run `python teleop/probe_fd.py` before
enabling force output.

## Optional datasets

Zarr recording and replay require:

```sh
python -m pip install --requirement teleop/requirements_dataset.txt
```

Generated datasets, videos, CSV files, and MuJoCo logs are intentionally ignored
by Git.
