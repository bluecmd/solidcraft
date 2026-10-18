"""Verify a generated study in a fresh private SolidCraft --control session.

Usage: SC_PORT=7880 SOLIDCRAFT_CONTROL_TOKEN=... python3 verify_study.py OUTPUT
OUTPUT must be accessible to this process and the app. Replaces this private
session's design with study.json; do not point it at a user's working window.
"""
import json
import math
import os
from pathlib import Path
import socket
import sys


def call(method, params):
    request = {"id": 1, "method": method, "params": params,
               "token": os.environ["SOLIDCRAFT_CONTROL_TOKEN"]}
    with socket.create_connection(("127.0.0.1", int(os.environ.get("SC_PORT", "7880"))), timeout=600) as connection:
        connection.sendall((json.dumps(request) + "\n").encode())
        chunks = bytearray()
        while not chunks.endswith(b"\n"):
            chunk = connection.recv(65536)
            if not chunk or len(chunks) + len(chunk) > 16 * 1024 * 1024:
                raise RuntimeError("missing or oversized control reply")
            chunks.extend(chunk)
    reply = json.loads(chunks)
    if not reply["ok"]:
        raise RuntimeError(reply.get("error", "control request failed"))
    return reply["result"]


def execute(command, params):
    return call("engine.execute", {"command": command, "params": params})


def fraction(step):
    x = min(max((step - 15) / 59, 0.0), 1.0)
    return 3 * x * x - 2 * x * x * x


def main():
    output = Path(sys.argv[1]).resolve()
    recipe = json.loads((output / "study.json").read_text())
    plan = json.loads((output / "plan.json").read_text())
    replies = call("engine.script", {"commands": recipe["commands"]})
    assert len(replies) == len(recipe["commands"])
    before, baseline, join_error = {}, {}, 0.0
    for command, reply in zip(recipe["commands"], replies):
        if command["command"] != "occurrence.properties":
            continue
        name = command["params"]["occurrence"]
        total = reply["total"]
        if name not in before:
            before[name] = total
        else:
            for key in ("min", "max"):
                join_error = max(join_error, *(abs(a - b) for a, b in zip(before[name]["bbox"][key], total["bbox"][key])))
            join_error = max(join_error, *(abs(a - b) for a, b in zip(before[name]["center_of_mass"], total["center_of_mass"])))
            baseline[name] = total
    assert len(baseline) == len(plan["parts"]) == 27 and join_error < 1e-6
    joints = execute("joint.list", {})
    assert not joints["conflicts"] and not joints.get("floating", [])
    assert 0 in joints["fixed"]
    assert all(row["dof"] == (0 if row["occurrence"] == 0 else 1) for row in joints["dof"])
    execute("motion.export", {"study": plan["study"], "samples": 90, "path": str(output / "poses.json")})
    rows = json.loads((output / "poses.json").read_text())["samples"]
    assert len(rows) == 90
    pose_error, checks = 0.0, 0
    identity = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    for row in rows:
        assert not row.get("stopped_by_loop") and not row.get("stopped_by_contact")
        poses = {p["occurrence"]: p["pose"] for p in row["positions"]}
        for part in plan["parts"]:
            pose = poses[part["occurrence"]]
            want = [fraction(row["step"]) * x for x in part["delta_mm"]]
            pose_error = max(pose_error, *(abs(a - b) for a, b in zip(pose["origin"], want)))
            for axis, expected in identity.items():
                assert all(abs(a - b) < 1e-12 for a, b in zip(pose[axis], expected))
            checks += 1
    assert pose_error < 1e-6
    bbox_error = 0.0
    for step in (0, 44, 89):
        played = execute("motion.play", {"study": plan["study"], "step": step})
        assert not played.get("unmoved") and not played.get("stopped_by_loop") and not played.get("stopped_by_contact")
        if step:
            assert played["moved"]
        for part in plan["parts"]:
            total = execute("occurrence.properties", {"occurrence": part["occurrence"]})["total"]
            original = baseline[part["occurrence"]]
            delta = [fraction(step) * x for x in part["delta_mm"]]
            for side in ("min", "max"):
                bbox_error = max(bbox_error, *(abs(a - b - d) for a, b, d in zip(total["bbox"][side], original["bbox"][side], delta)))
            assert math.isclose(total["volume_mm3"], original["volume_mm3"], rel_tol=1e-10, abs_tol=1e-6)
            assert math.isclose(total["mass_g"], original["mass_g"], rel_tol=1e-10, abs_tol=1e-6)
    assert bbox_error < 1e-6
    scene = execute("render.scene", {})
    ordinals = {}
    for index, body in enumerate(scene["bodies"]):
        assert body["name"] not in ordinals
        # #2067's earlier render.scene used the ordered array without the
        # newer explicit ordinal field. Both are captured scene indices.
        ordinals[body["name"]] = body.get("ordinal", body.get("index", index))

    def bind(value):
        if isinstance(value, dict):
            if "body_index" in value and "body_name" in value:
                value["body_index"] = ordinals[value["body_name"]]
            for child in value.values():
                bind(child)
        elif isinstance(value, list):
            for child in value:
                bind(child)

    for template in output.glob("render-template-*.json"):
        params = json.loads(template.read_text())
        bind(params)
        (output / template.name.replace("render-template-", "render-")).write_text(json.dumps(params, indent=2) + "\n")
    execute("motion.play", {"study": plan["study"], "step": 0})
    execute("file.save_as", {"path": str(output / "coffee-study.solidcraft")})
    report = {"passed": True, "parts": len(plan["parts"]), "frames": 90, "pose_checks": checks,
              "join_error_mm": join_error, "pose_error_mm": pose_error, "bbox_error_mm": bbox_error,
              "conflicts": [], "floating": [], "saved_step": 0, "scene_bindings": ordinals,
              "application": call("ui.help", {})}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("scene_bindings", "application")}, indent=2))


if __name__ == "__main__":
    main()
