"""
Force Dimension SDK - Python (ctypes) binding.

Adapted from the official binding shipped with the Force Dimension SDK 3.17.7
(``extensions/Python/fdsdk.py``). The shipped file left the library-loading
line commented out (so ``Init()`` did nothing and every call failed); here
``Init()`` actually loads ``libdrd.so.3``. A couple of ``restype``
declarations are added for 64-bit pointer correctness.

The library path defaults to this machine's SDK location but can be overridden
with the ``FORCEDIMENSION_LIB`` environment variable.
"""

import ctypes
import ctypes.util
import os
from pathlib import Path

import numpy as np

def _default_library():
    """Find libdrd in the loader path or common unpacked-SDK locations."""
    system_library = ctypes.util.find_library("drd")
    if system_library:
        return system_library
    home = Path.home()
    search_roots = (
        home / "Documents" / "Force Dimension",
        home / "Downloads",
    )
    for root in search_roots:
        matches = sorted(
            root.glob("sdk-*/lib/release/lin-x86_64-gcc/libdrd.so.3"),
            reverse=True,
        )
        if matches:
            # Prefer Documents over Downloads and the newest version in each.
            return str(matches[0])
    return "libdrd.so.3"


# libdrd re-exports the dhd* symbols. FORCEDIMENSION_LIB remains the explicit
# override; otherwise discover the system library or an unpacked SDK.
_DEFAULT_LIB = _default_library()

libdrd = None


def Init(lib_path=None):
    """Load the Force Dimension shared library. Must be called before anything else."""
    global libdrd
    lib_path = lib_path or os.environ.get("FORCEDIMENSION_LIB", _DEFAULT_LIB)
    try:
        libdrd = ctypes.CDLL(lib_path)
    except OSError as error:
        raise FileNotFoundError(
            f"Force Dimension library could not be loaded from '{lib_path}'. "
            "Install the Force Dimension SDK or set FORCEDIMENSION_LIB to "
            "libdrd.so.3."
        ) from error
    # dhdGetSystemName returns a char* -> must not be truncated to int on 64-bit.
    libdrd.dhdGetSystemName.restype = ctypes.c_char_p
    # these return C bool
    libdrd.dhdHasWrist.restype = ctypes.c_bool
    libdrd.dhdHasActiveGripper.restype = ctypes.c_bool
    libdrd.dhdHasActiveWrist.restype = ctypes.c_bool
    libdrd.dhdHasGripper.restype = ctypes.c_bool
    return libdrd


def HasActiveGripper(id):
    """True on devices with a force-feedback gripper (omega.7)."""
    return bool(libdrd.dhdHasActiveGripper(id))


def HasWrist(id):
    """True when the device can report wrist orientation."""
    return bool(libdrd.dhdHasWrist(id))


def HasActiveWrist(id):
    return bool(libdrd.dhdHasActiveWrist(id))


def Open():
    """Open a connection to the first haptic device. Returns device id (>=0) or -1."""
    return libdrd.drdOpen()


def EnableForce(on, id):
    return libdrd.dhdEnableForce(on, id)


def SetBrakes(on, id):
    """Engage (1) / release (0) the electromagnetic brakes (viscous damping)."""
    return libdrd.dhdSetBrakes(on, id)


def EmulateButton(on, id):
    return libdrd.dhdEmulateButton(on, id)


def IsInitialized(id):
    return libdrd.drdIsInitialized(id)


def AutoInit(id):
    return libdrd.drdAutoInit(id)


def Start(id):
    return libdrd.drdStart(id)


def Stop(keepForcesOn, id):
    return libdrd.drdStop(keepForcesOn, id)


def MoveToPos(px, py, pz, block, id):
    """Move the base to a Cartesian position under robotic regulation.
    Requires regulation to be running (after AutoInit or Start)."""
    return libdrd.drdMoveToPos(
        ctypes.c_double(px), ctypes.c_double(py), ctypes.c_double(pz),
        ctypes.c_bool(block), id,
    )


def GetPosMoveParam(id):
    amax = ctypes.c_double(0.0)
    vmax = ctypes.c_double(0.0)
    jerk = ctypes.c_double(0.0)
    ret = libdrd.drdGetPosMoveParam(
        ctypes.pointer(amax), ctypes.pointer(vmax), ctypes.pointer(jerk), id
    )
    return ret, amax.value, vmax.value, jerk.value


def SetPosMoveParam(amax, vmax, jerk, id):
    return libdrd.drdSetPosMoveParam(
        ctypes.c_double(amax), ctypes.c_double(vmax), ctypes.c_double(jerk), id
    )


def GetSystemType(id):
    return libdrd.dhdGetSystemType(id)


def GetSystemName(id):
    raw = libdrd.dhdGetSystemName(id)  # bytes (restype set to c_char_p)
    return raw.decode("utf-8") if raw else ""


def GetSerialNumber(id):
    sn = ctypes.c_ushort(0)
    libdrd.dhdGetSerialNumber(ctypes.pointer(sn), id)
    return sn.value


def GetPosition(id):
    x = ctypes.c_double(0.0)
    y = ctypes.c_double(0.0)
    z = ctypes.c_double(0.0)
    ret = libdrd.dhdGetPosition(
        ctypes.pointer(x), ctypes.pointer(y), ctypes.pointer(z), id
    )
    return ret, x.value, y.value, z.value


def GetOrientationFrame(id):
    rotation = np.zeros((3, 3), dtype=np.double)
    ret = libdrd.dhdGetOrientationFrame(
        rotation.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), id
    )
    return ret, rotation


def GetGripperGap(id):
    g = ctypes.c_double(0.0)
    ret = libdrd.dhdGetGripperGap(ctypes.pointer(g), id)
    return ret, g.value


def GetForce(id):
    fx = ctypes.c_double(0.0)
    fy = ctypes.c_double(0.0)
    fz = ctypes.c_double(0.0)
    ret = libdrd.dhdGetForce(
        ctypes.pointer(fx), ctypes.pointer(fy), ctypes.pointer(fz), id
    )
    return ret, fx.value, fy.value, fz.value


def GetLinearVelocity(id):
    vx = ctypes.c_double(0.0)
    vy = ctypes.c_double(0.0)
    vz = ctypes.c_double(0.0)
    ret = libdrd.dhdGetLinearVelocity(
        ctypes.pointer(vx), ctypes.pointer(vy), ctypes.pointer(vz), id
    )
    return ret, vx.value, vy.value, vz.value


def GetGripperLinearVelocity(id):
    vg = ctypes.c_double(0.0)
    ret = libdrd.dhdGetGripperLinearVelocity(ctypes.pointer(vg), id)
    return ret, vg.value


def SetForce(fx, fy, fz, id):
    return libdrd.dhdSetForce(
        ctypes.c_double(fx), ctypes.c_double(fy), ctypes.c_double(fz), id
    )


def SetForceAndTorqueAndGripperForce(fx, fy, fz, tx, ty, tz, fg, id):
    return libdrd.dhdSetForceAndTorqueAndGripperForce(
        ctypes.c_double(fx),
        ctypes.c_double(fy),
        ctypes.c_double(fz),
        ctypes.c_double(tx),
        ctypes.c_double(ty),
        ctypes.c_double(tz),
        ctypes.c_double(fg),
        id,
    )


def GetButton(index, id):
    return libdrd.dhdGetButton(index, id)


def Close(id):
    return libdrd.dhdClose(id)
