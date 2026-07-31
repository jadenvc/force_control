"""
Safe headless probe of the Force Dimension omega from Python.

Opens the device, prints its identity and calibration status, and -- if it is
already calibrated -- streams the live pose for a few seconds. Does NOT move
the device unless --auto-init is passed. Use this to confirm the Python
binding + USB access work before running the full teleop.

    python probe_fd.py            # read-only, no movement
    python probe_fd.py --auto-init --seconds 3
"""

import argparse
import time

import fdsdk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto-init", action="store_true",
                    help="auto-calibrate if needed (DEVICE WILL MOVE)")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    fdsdk.Init()
    print("library loaded OK")

    dev = fdsdk.Open()
    if dev < 0:
        print("ERROR: no device found (powered on? held by another program?)")
        return 1
    print(f"opened device id={dev}")
    print(f"system : {fdsdk.GetSystemName(dev)} (type {fdsdk.GetSystemType(dev)})")
    print(f"serial : {fdsdk.GetSerialNumber(dev)}")
    print(f"gripper: {'active (omega.7)' if fdsdk.HasActiveGripper(dev) else 'none (omega.6)'}")

    initialized = bool(fdsdk.IsInitialized(dev))
    print(f"calibrated: {initialized}")

    if not initialized:
        if args.auto_init:
            print(">>> auto-calibrating (device will move, keep clear)...")
            fdsdk.AutoInit(dev)
            fdsdk.Stop(True, dev)
            print(">>> done")
            initialized = True
        else:
            print("not calibrated -> skipping live read. "
                  "Re-run with --auto-init (device will move) or run bin/HapticInit.")

    if initialized:
        fdsdk.EnableForce(1, dev)
        print(f"\nstreaming pose for {args.seconds:.1f}s "
              f"(move the handle; press user button to see it flip):")
        t_end = time.time() + args.seconds
        while time.time() < t_end:
            _, x, y, z = fdsdk.GetPosition(dev)
            _, gap = fdsdk.GetGripperGap(dev)
            btn = fdsdk.GetButton(0, dev)
            fdsdk.SetForce(0.0, 0.0, 0.0, dev)  # float under gravity comp
            print(f"  pos=({x:+.4f}, {y:+.4f}, {z:+.4f}) m   "
                  f"gripper={gap:+.4f}   button={btn}   ", end="\r")
            time.sleep(0.05)
        print()

    fdsdk.Close(dev)
    print("closed. OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
