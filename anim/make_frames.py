"""Per-frame recipes and render requests for an assembled -> exploded animation.

Each frame appends rigid solid.move translations to the original exploded
recipe: assembled offset + s * (framing offset - assembled offset), s eased.
"""
import copy, json, math, sys
from pathlib import Path

STUDIO = Path(sys.argv[1])           # examples/studio
OUT = Path(sys.argv[2])
FRAMES = 90
sys.path.insert(0, str(STUDIO))
from author_coffee_framing import SHIFTS, CAMERA   # framing (exploded end state)

# Hand-placed assembled pose, as offsets from the original exploded recipe.
# Glass (tank, bin) stays clear of every other body.
ASSEMBLED = {
    'Open brewing head': [0, 0, 0], 'Brewing collar': [0, 0, 0], 'Shower screen': [0, 0, 0],
    'Brewing head roof': [0, 0, -2], 'Bent chrome lever': [0, 0, -2],
    'Left lever ball': [0, 0, -2], 'Right lever ball': [0, 0, -2],
    'Smoky D tank': [-95, 0, 0], 'Separate tank lid': [-95, 0, -32], 'Tank lid grip': [-95, 0, -32],
    'Clear ribbed bin': [45, 0, 0],
    'Louvred service panel': [122, -74, 0],
    'Drip tray': [0, 80, 0], 'Perforated stainless grate': [0, 80, -27], 'Rounded tray cap': [-60, 80, 0],
    'Black S power cable': [-40, 60, 0], 'Two pin plug': [-60, 70, 0],
    'Plug pin 206': [-60, 70, 0], 'Plug pin 218': [-60, 70, 0],
}
for i in range(1, 9):
    ASSEMBLED[f'Zinc screw {i}'] = [30 if i <= 4 else -30, 0, 0]

def ease(x):
    return x * x * (3 - 2 * x)

recipe = json.loads((STUDIO / 'coffee-exploded-sweep.json').read_text())
render = json.loads((STUDIO / 'coffee-framing-render.json').read_text())
OUT.mkdir(parents=True, exist_ok=True)
names = set(ASSEMBLED) | set(SHIFTS)
for f in range(FRAMES):
    # Hold assembled 15 frames, explode over 60, hold exploded 15.
    s = ease(min(max((f - 15) / 59, 0.0), 1.0))
    r = copy.deepcopy(recipe)
    for name in sorted(names):
        a = ASSEMBLED.get(name, [0, 0, 0]); e = SHIFTS.get(name, [0, 0, 0])
        t = [a[k] + s * (e[k] - a[k]) for k in range(3)]
        if any(abs(v) > 1e-9 for v in t):
            r['commands'].append(dict(command='solid.move', params=dict(bodies=[name], translate=t, copy=False, name='Animation: ' + name)))
    q = copy.deepcopy(render)
    cam = dict(CAMERA)
    cam['yaw'] = CAMERA['yaw'] + math.radians(12) * (f / (FRAMES - 1) - 0.5)
    q['settings']['camera'] = cam
    q['settings']['width'] = 1280; q['settings']['height'] = 720
    q['samples'] = 64
    q['denoise'] = True
    (OUT / f'recipe-{f:03d}.json').write_text(json.dumps(r))
    (OUT / f'render-{f:03d}.json').write_text(json.dumps(q))
print('wrote', FRAMES, 'frames to', OUT)
