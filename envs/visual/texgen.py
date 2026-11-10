"""Procedural texture pools for StationeryBench sim visuals (deterministic, regenerated on demand).

python texgen.py OUTDIR           # writes OUTDIR/{nominal,train,heldout}/*.jpg|png + pool.json

Wood textures are laid out for the table-top render box built in visual.py.  Measured SAPIEN box
top-face UV (probe2): u (image columns) runs along box local x, v (image rows) along local +y.  The
wood is generated as a 2:1 image with grain along columns and then TRANSPOSED on save, so the saved
image is 1536 (u, world x, 1.209 m) x 3072 (v, world y, 2.418 m) -> isotropic 1.27 px/mm and the
grain runs along world y (left-right in the operator photo).

Nominal wood is tuned to the Robocurve bench photo (docs/assets/*-setup.jpg): dark, worn, low-
saturation brown with lighter worn patches, darker knots/blotches, fine grain, one plank seam
along x (operator-left of centre) and one along y near the arm bases.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

WOOD_W, WOOD_H = 3072, 1536          # px; 2.418 m x 1.209 m  -> 1.27 px/mm
TABLE_Y0, TABLE_Y1 = -1.2149, 1.2030  # table aabb (world y) -> u = (y1 - y) / (y1 - y0) ... see visual.py
TABLE_X0, TABLE_X1 = -0.7402, 0.4689


def _up(g, W, H):
    return np.asarray(Image.fromarray(g.astype(np.float32), mode="F").resize((W, H), Image.BICUBIC))


def fbm(rng, H, W, octaves):
    """octaves: list of (cell_h, cell_w, amp) -> sum of bicubic-upsampled gaussian grids."""
    out = np.zeros((H, W), np.float32)
    for ch, cw, a in octaves:
        g = rng.standard_normal((max(2, H // ch + 2), max(2, W // cw + 2))).astype(np.float32)
        out += a * _up(g, W + 2 * cw, H + 2 * ch)[ch:ch + H, cw:cw + W]
    return out


def srgb2lin(c):
    c = np.asarray(c, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055) * 255.0


def wood(seed, dark, mid, wear_col, grain=1.0, ring_period=22.0, wear=0.35, knots=6,
         seam_cols=(0.36,), seam_rows=(0.10,), plank_tone=0.10, scratches=40, W=WOOD_W, H=WOOD_H):
    """Return uint8 HxWx3 sRGB wood texture.  dark/mid/wear_col are sRGB triplets."""
    rng = np.random.default_rng(seed)
    fine = fbm(rng, H, W, [(2, 48, 1.0), (3, 120, 0.8), (6, 300, 0.6)])
    warp = fbm(rng, H, W, [(80, 400, 1.0), (40, 160, 0.4)])
    rows = np.arange(H, dtype=np.float32)[:, None]
    rings = np.sin(2 * np.pi * (rows / ring_period + 2.5 * warp))
    rings = np.sign(rings) * np.abs(rings) ** 0.5
    blotch = fbm(rng, H, W, [(200, 260, 1.0), (90, 140, 0.6), (30, 60, 0.3)])
    v = 0.28 * grain * fine + 0.16 * grain * rings + 0.28 * blotch
    # planks: per-board tone offset (boards split by the seams)
    cols_edges = [0] + [int(c * W) for c in seam_cols] + [W]
    rows_edges = [0] + [int(r * H) for r in seam_rows] + [H]
    tone = np.zeros((H, W), np.float32)
    for i in range(len(rows_edges) - 1):
        for j in range(len(cols_edges) - 1):
            tone[rows_edges[i]:rows_edges[i + 1], cols_edges[j]:cols_edges[j + 1]] = rng.normal(0, plank_tone)
    v = v + tone / 0.25
    t = 1 / (1 + np.exp(-1.6 * v))                                    # 0 dark .. 1 mid
    d, m, w = srgb2lin(dark), srgb2lin(mid), srgb2lin(wear_col)
    col = d[None, None] * (1 - t[..., None]) + m[None, None] * t[..., None]
    # worn / sanded patches: lighter, greyer, with grain still showing through
    wn = fbm(rng, H, W, [(60, 160, 1.0), (25, 70, 0.5), (8, 30, 0.25)])
    wa = np.clip((wn - (1.6 - 2.0 * wear)) * 0.9, 0, 1) * (0.45 + 0.55 * (0.5 + 0.5 * fine.clip(-1, 1)))
    col = col * (1 - 0.45 * wa[..., None]) + w[None, None] * 0.45 * wa[..., None]
    # dark knots / water marks: elongated along the grain
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dk = np.zeros((H, W), np.float32)
    for _ in range(knots):
        cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        ry, rx = rng.uniform(4, 16), rng.uniform(12, 70)
        dk = np.maximum(dk, np.exp(-(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)) * rng.uniform(0.4, 0.8))
    streak = np.clip(fbm(rng, H, W, [(3, 200, 1.0)]) - 1.3, 0, 1) * 0.6     # long dark streaks
    col = col * (1 - np.maximum(dk, streak)[..., None] * 0.65)
    # small light chips (worn-through spots, like the beige chip in the photo)
    for _ in range(max(1, knots // 6)):
        cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        ry, rx = rng.uniform(5, 14), rng.uniform(10, 30)
        b = np.exp(-(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) ** 2)
        b = np.clip(b * 2 - 0.5, 0, 1) * (fbm(rng, H, W, [(6, 6, 1.0)]) > -0.8)
        col = col * (1 - b[..., None]) + srgb2lin((190, 170, 140))[None, None] * b[..., None]
    img = lin2srgb(col)
    im = Image.fromarray(img.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for _ in range(scratches):                                             # faint light scratches
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
        ang, ln = rng.uniform(0, np.pi), rng.uniform(30, 260)
        c = tuple(int(x) for x in np.clip(np.array(mid) * 1.08 + 8, 0, 255))
        dr.line([(x0, y0), (x0 + ln * np.cos(ang), y0 + ln * np.sin(ang))], fill=c, width=1)
    for c in seam_cols:                                                    # plank seams
        x = int(c * W); dr.line([(x, 0), (x, H)], fill=(20, 15, 12), width=2)
    for r in seam_rows:
        y = int(r * H); dr.line([(0, y), (W, y)], fill=(20, 15, 12), width=2)
    return np.asarray(im)


def flat_noise(seed, base, amp=6.0, W=1024, H=512, smudge=0.0):
    """Backdrop / floor texture: base sRGB colour + mild noise (+ optional smudges)."""
    rng = np.random.default_rng(seed)
    n = fbm(rng, H, W, [(64, 64, 1.0), (8, 8, 0.4), (2, 2, 0.3)])
    img = np.array(base, np.float32)[None, None] + amp * n[..., None]
    if smudge > 0:
        s = np.clip(fbm(rng, H, W, [(40, 40, 1.0), (12, 12, 0.5)]) - 0.8, 0, 2)
        img = img + smudge * s[..., None] * np.array([0.8, 0.9, 1.0])
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- pools
NOMINAL_WOOD = dict(seed=1001, dark=(50, 42, 37), mid=(108, 94, 84), wear_col=(150, 142, 134), grain=1.0,
                    ring_period=22.0, wear=0.12, knots=18, seam_cols=(0.38,), seam_rows=(0.10,))


def wood_pool(split):
    """Train: 12 woods spanning walnut..oak..grey-wash; heldout: 4 disjoint seeds AND palettes
    (reddish mahogany, cold grey-brown, light honey, near-black ebony) never used in training."""
    out = []
    if split == "train":
        pals = [((50, 36, 28), (108, 84, 66), (148, 132, 118)), ((40, 30, 24), (92, 72, 58), (130, 120, 110)),
                ((62, 44, 30), (130, 98, 70), (168, 150, 126)), ((45, 36, 32), (100, 86, 76), (140, 134, 128)),
                ((70, 50, 34), (140, 106, 76), (176, 156, 130)), ((36, 28, 24), (82, 64, 52), (120, 110, 100)),
                ((58, 40, 26), (122, 90, 62), (160, 140, 116)), ((48, 40, 36), (110, 96, 86), (150, 146, 140)),
                ((66, 46, 36), (126, 92, 72), (160, 140, 124)), ((54, 42, 30), (116, 96, 74), (150, 140, 120)),
                ((44, 32, 26), (100, 76, 60), (140, 126, 112)), ((60, 48, 38), (128, 108, 88), (170, 160, 146))]
        for i, (d, m, w) in enumerate(pals):
            rs = np.random.default_rng(100 + i)
            out.append(dict(seed=2000 + i, dark=d, mid=m, wear_col=w, grain=float(rs.uniform(0.7, 1.3)),
                            ring_period=float(rs.uniform(14, 34)), wear=float(rs.uniform(0.05, 0.40)),
                            knots=int(rs.integers(6, 24)), seam_cols=(float(rs.uniform(0.2, 0.45)),),
                            seam_rows=(float(rs.uniform(0.05, 0.15)),)))
    else:
        pals = [((70, 30, 22), (140, 64, 44), (170, 120, 100)), ((40, 40, 42), (96, 94, 92), (150, 150, 150)),
                ((90, 64, 36), (176, 136, 90), (200, 180, 150)), ((22, 18, 16), (54, 44, 38), (96, 90, 84))]
        for i, (d, m, w) in enumerate(pals):
            rs = np.random.default_rng(900 + i)
            out.append(dict(seed=9000 + i, dark=d, mid=m, wear_col=w, grain=float(rs.uniform(0.7, 1.3)),
                            ring_period=float(rs.uniform(14, 34)), wear=float(rs.uniform(0.05, 0.40)),
                            knots=int(rs.integers(6, 24)), seam_cols=(float(rs.uniform(0.55, 0.8)),),
                            seam_rows=(float(rs.uniform(0.05, 0.15)),)))
    return out


NOMINAL_BACKDROP = dict(seed=1101, base=(10, 11, 14), amp=3.0, smudge=18.0)   # glossy black panel, smudges
BACKDROP_POOL = {
    "train": [dict(seed=2100 + i, base=b, amp=a, smudge=s) for i, (b, a, s) in enumerate([
        ((6, 6, 8), 2, 10), ((20, 20, 22), 4, 0), ((30, 26, 22), 5, 0), ((14, 18, 26), 4, 12),
        ((45, 45, 48), 6, 0), ((25, 22, 30), 4, 8), ((60, 58, 54), 8, 0), ((3, 3, 3), 1, 25),
        ((120, 118, 112), 8, 0), ((36, 40, 36), 5, 0)])],
    "heldout": [dict(seed=9100 + i, base=b, amp=a, smudge=s) for i, (b, a, s) in enumerate([
        ((18, 10, 10), 4, 6), ((90, 96, 110), 8, 0), ((200, 198, 190), 6, 0), ((10, 22, 14), 3, 10)])],
}


def build_all(out):
    pool = {"nominal": {}, "train": {}, "heldout": {}}
    os.makedirs(f"{out}/nominal", exist_ok=True)
    Image.fromarray(wood(**NOMINAL_WOOD).transpose(1, 0, 2).copy()).save(f"{out}/nominal/wood.jpg", quality=93)
    Image.fromarray(flat_noise(**NOMINAL_BACKDROP)).save(f"{out}/nominal/backdrop.png")
    pool["nominal"] = dict(wood="nominal/wood.jpg", backdrop="nominal/backdrop.png",
                           wood_params=NOMINAL_WOOD, backdrop_params=NOMINAL_BACKDROP)
    for split in ("train", "heldout"):
        os.makedirs(f"{out}/{split}", exist_ok=True)
        woods, bds = [], []
        for i, p in enumerate(wood_pool(split)):
            fn = f"{split}/wood_{i:02d}.jpg"
            Image.fromarray(wood(**p).transpose(1, 0, 2).copy()).save(f"{out}/{fn}", quality=93); woods.append(fn)
            print(fn, flush=True)
        for i, p in enumerate(BACKDROP_POOL[split]):
            fn = f"{split}/backdrop_{i:02d}.png"
            Image.fromarray(flat_noise(**p)).save(f"{out}/{fn}"); bds.append(fn)
        pool[split] = dict(wood=woods, backdrop=bds,
                           wood_params=wood_pool(split), backdrop_params=BACKDROP_POOL[split])
    json.dump(pool, open(f"{out}/pool.json", "w"), indent=1, default=list)


if __name__ == "__main__":
    build_all(sys.argv[1])
