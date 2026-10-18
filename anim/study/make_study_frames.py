"""Author the original coffee explosion as native joints and one motion study.

Usage: python3 make_study_frames.py SOLIDCRAFT/examples/studio OUTPUT
Only the initial assembled geometry uses solid.move. Every animated frame is
selected by motion.play; the study also works with motion.export_animation.
"""
import copy
import argparse
import json
import math
from pathlib import Path
import sys

FRAMES = 90
STUDY = "Coffee explosion"
ASSEMBLED = {
    "Open brewing head": [0, 0, 0], "Brewing collar": [0, 0, 0], "Shower screen": [0, 0, 0],
    "Brewing head roof": [0, 0, -2], "Bent chrome lever": [0, 0, -2],
    "Left lever ball": [0, 0, -2], "Right lever ball": [0, 0, -2],
    "Smoky D tank": [-95, 0, 0], "Separate tank lid": [-95, 0, -32], "Tank lid grip": [-95, 0, -32],
    "Clear ribbed bin": [45, 0, 0], "Louvred service panel": [122, -74, 0],
    "Drip tray": [0, 80, 0], "Perforated stainless grate": [0, 80, -27], "Rounded tray cap": [-60, 80, 0],
    "Black S power cable": [-40, 60, 0], "Two pin plug": [-60, 70, 0],
    "Plug pin 206": [-60, 70, 0], "Plug pin 218": [-60, 70, 0],
}
ASSEMBLED.update({f"Zinc screw {i}": [30 if i <= 4 else -30, 0, 0] for i in range(1, 9)})


def fraction(frame):
    x = min(max((frame - 15) / 59, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studio", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--samples", type=int, default=512)
    args = parser.parse_args()
    if not (16 <= args.width <= 3840 and 16 <= args.height <= 2160 and 1 <= args.samples <= 4096):
        parser.error("dimensions or sample count outside authoring limits")
    studio, output = args.studio, args.output
    sys.path.insert(0, str(studio.resolve()))
    from author_coffee_framing import SHIFTS, CAMERA

    output.mkdir(parents=True, exist_ok=True)
    recipe = json.loads((studio / "coffee-exploded-sweep.json").read_text())
    render = json.loads((studio / "coffee-framing-render.json").read_text())
    names = sorted(set(ASSEMBLED) | set(SHIFTS))
    commands = recipe["commands"]
    # Static modelling placement only: geometry is authored in the assembled pose.
    for name in names:
        initial = ASSEMBLED.get(name, [0, 0, 0])
        if any(initial):
            commands.append({"command": "solid.move", "params": {
                "bodies": [name], "translate": initial, "copy": False, "name": "Assembled: " + name,
            }})

    parts, keys = [], []
    for name in names:
        initial = ASSEMBLED.get(name, [0, 0, 0])
        end = SHIFTS.get(name, [0, 0, 0])
        delta = [b - a for a, b in zip(initial, end)]
        travel = math.sqrt(sum(x * x for x in delta))
        if travel == 0:
            continue
        component = "Motion " + name
        joint = "Explode " + name
        axis = [x / travel for x in delta]
        commands.extend([
            {"command": "component.create", "params": {"name": component, "parent": "root", "activate": False}},
            {"command": "component.move_bodies", "params": {"bodies": [name], "component": component}},
            {"command": "occurrence.properties", "params": {"occurrence": component + ":1"}},
            {"command": "joint.as_built", "params": {
                "type": "slider", "a": "root", "b": component,
                "at": {"occurrence": "root", "point": [0, 0, 0], "z": axis},
                "name": joint, "limits": [[0, travel]], "quiet": True,
            }},
            {"command": "occurrence.properties", "params": {"occurrence": component + ":1"}},
        ])
        keys.append({"joint": joint, "points": [[f, fraction(f) * travel] for f in range(FRAMES)]})
        parts.append({"body": name, "occurrence": component + ":1", "joint": joint,
                      "axis": axis, "travel_mm": travel, "delta_mm": delta,
                      "assembled_offset_mm": initial, "exploded_offset_mm": end})
    commands.append({"command": "motion.study", "params": {"name": STUDY, "steps": FRAMES - 1, "keys": keys}})
    saved = copy.deepcopy(recipe)
    saved["commands"].append({"command": "motion.play", "params": {"study": STUDY, "step": 0}})
    write(output / "study.json", saved)
    write(output / "plan.json", {
        "study": STUDY, "steps": FRAMES - 1, "grounded": "root (0)",
        "joints": "independent as-built sliders from grounded root; no loops or links",
        "initial_pose": "assembled", "saved_pose": 0, "parts": parts,
        "expected_occurrence_translation": "fraction(step) * delta_mm; orientation remains identity",
    })
    for frame in range(FRAMES):
        r = copy.deepcopy(recipe)
        # Measurement commands belong to one-time validation, not each render.
        r["commands"] = [c for c in r["commands"] if c["command"] != "occurrence.properties"]
        r["commands"].append({"command": "motion.play", "params": {"study": STUDY, "step": frame}})
        r["commands"].append({"command": "joint.list", "params": {}})
        q = copy.deepcopy(render)
        camera = dict(CAMERA)
        camera["yaw"] += math.radians(12) * (frame / (FRAMES - 1) - 0.5)
        q["settings"].update(camera=camera, width=args.width, height=args.height)
        q.update(samples=args.samples, denoise=True)
        (output / f"render-{frame:03d}.json").unlink(missing_ok=True)
        write(output / f"recipe-{frame:03d}.json", r)
        # Component placement can change captured scene ordinals. The verifier
        # binds these templates to the exact render.scene names and ordinals.
        write(output / f"render-template-{frame:03d}.json", q)
    print(f"Wrote {len(parts)} grounded sliders, one study and {FRAMES} motion.play frames to {output}")


if __name__ == "__main__":
    main()
