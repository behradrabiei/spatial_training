"""Reset a running LightNav Habitat env server once and print what it serves (validate step).

Runs in the LightNav client env (needs pyzmq + numpy only). Sends ``reset`` and ``close``
through the same pickle-over-ZMQ protocol as ``lightnav.habitat.remote_env``.
"""

from __future__ import annotations

import argparse
import pickle
import sys

import zmq


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="tcp://localhost:5555")
    p.add_argument("--timeout-ms", type=int, default=600000)
    p.add_argument("--no-close", action="store_true")
    a = p.parse_args(argv)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, a.timeout_ms)
    sock.connect(a.server)

    def call(command, data=None):
        sock.send(pickle.dumps({"command": command, "data": data}, protocol=4))
        resp = pickle.loads(sock.recv())
        if resp.get("status") != "success":
            raise RuntimeError(f"{command}: {resp.get('message')}")
        return resp

    resp = call("reset", {"seed": None, "options": None})
    obs, info = resp["obs"], resp["info"]
    rgb = obs["rgb"]
    print("rgb", getattr(rgb, "shape", None), getattr(rgb, "dtype", None))
    print("instruction", obs.get("instruction"))
    keys = ("episode_id", "raw_episode_id", "scene_id", "object_category", "distance_to_goal",
            "habitat_time_step", "lin_vel_range", "ang_vel_range")
    for k in keys:
        print(f"info[{k!r}] = {info.get(k)!r}")
    ok = tuple(rgb.shape) == (270, 480, 3) and all(k in info for k in keys)
    if not a.no_close:
        call("close")
    print("probe", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
