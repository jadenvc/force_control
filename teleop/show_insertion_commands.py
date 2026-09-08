"""Print the commands/args saved for an insertion dataset, so a session's
settings can actually be found again later.

Two independent sources, both populated automatically by teleop_insertion.py
(see that file's ``_log_session_command`` and the ``metadata={"command_line":
...}`` passed to ``InsertionEpisodeRecorder.start_episode``):

  1. ``<dataset>.sessions.jsonl`` -- one line per script invocation that used
     --collect-dataset, written at startup regardless of what happens
     afterward (episodes kept, discarded, or the operator quits without
     starting one). This is the only source if a session never kept an
     episode.
  2. Each KEPT episode's own ``metadata_json`` attr (inside the zarr store) --
     the args that were active when THAT SPECIFIC episode was recorded. Two
     episodes in the same dataset can have different args if the operator
     restarted the script with different flags mid-dataset; the session log
     alone can't tell you which episode used which invocation, only the
     per-episode metadata can.

Usage:
    python show_insertion_commands.py ~/data/insertion_v2.zarr                  # both sources
    python show_insertion_commands.py ~/data/insertion_v2.zarr --sessions-only
    python show_insertion_commands.py ~/data/insertion_v2.zarr --episodes-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _command_from_args(args: dict) -> str:
    """Reconstruct a runnable command line from a saved args dict.

    Best-effort, not byte-exact: argparse's ``vars(args)`` loses the
    distinction between "flag omitted" and "flag explicitly set to its
    default", so this reconstruction includes every non-default-looking
    value it can, on the reasonable assumption that a saved run is more
    useful over-specified than under-specified.
    """
    parts = ["python", "teleop_insertion.py"]
    skip = {"collect_dataset"}  # path is dataset-specific, not part of the "settings" a user re-runs
    for key, value in sorted(args.items()):
        if key in skip or value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            parts.append(flag)
        elif isinstance(value, (list, tuple)):
            parts.append(flag)
            parts.extend(str(v) for v in value)
        else:
            parts.append(flag)
            parts.append(str(value))
    return " ".join(parts)


def show_sessions(dataset_path: Path) -> None:
    log_path = dataset_path.with_suffix(dataset_path.suffix + ".sessions.jsonl")
    if not log_path.exists():
        print(f"(no session log at {log_path})")
        return
    print(f"=== session log: {log_path} ===")
    with open(log_path) as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            print(f"\n[session {i}] started {entry['started_utc']}")
            print(f"  {entry['command']}")


def show_episodes(dataset_path: Path) -> None:
    import zarr

    root = zarr.open(str(dataset_path), mode="r")
    names = sorted(root["data"].group_keys(), key=lambda k: int(k.rsplit("_", 1)[-1]))
    if not names:
        print("(no kept episodes)")
        return
    print(f"\n=== per-episode saved args: {dataset_path} ===")
    for name in names:
        ep = root["data"][name]
        meta_raw = ep.attrs.get("metadata_json")
        print(f"\n[{name}] success={ep.attrs.get('success')} "
              f"broken={ep.attrs.get('broken')} reason={ep.attrs.get('termination_reason')}")
        if not meta_raw:
            print("  (no saved command_line metadata for this episode)")
            continue
        meta = json.loads(meta_raw)
        args = meta.get("command_line")
        if not args:
            print("  (metadata present but no command_line key)")
            continue
        print(f"  {_command_from_args(args)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path")
    parser.add_argument("--sessions-only", action="store_true")
    parser.add_argument("--episodes-only", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not args.episodes_only:
        show_sessions(dataset_path)
    if not args.sessions_only:
        show_episodes(dataset_path)


if __name__ == "__main__":
    main()
