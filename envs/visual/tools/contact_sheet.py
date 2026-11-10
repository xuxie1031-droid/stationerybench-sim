"""contact_sheet.py LOOKDIR TASKNAME REALPHOTO OUT.jpg  (LOOKDIR from look_test.py)"""
import os, sys  # noqa: E401  repo root on sys.path -> gyre_paths (scripts/env.sh does the same)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import gyre_paths as GP  # noqa: E402
import sys
from PIL import Image, ImageDraw, ImageFont
look, task, photo, out = sys.argv[1:5]
REAL_TOP = os.path.join(GP.VISUAL, "real_ref/real_top_t5.png")
REAL_WRIST = os.path.join(GP.VISUAL, "real_ref/real_left_t5.png")
cols = [("real", None), ("old sim", "old"), ("nominal", "nominal"), ("DR train 0", "dr0"), ("DR train 1", "dr1"),
        ("DR train 2", "dr2"), ("DR held-out", "ho0")]
rows = [("top_cam", REAL_TOP, "real top_cam: AI2 YAM bench (other table)"),
        ("left_cam", REAL_WRIST, "real left wrist: AI2 YAM bench"),
        ("operator", photo, "real: Robocurve setup photo")]
W, H, LH = 400, 225, 22
try:
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except Exception:
    f = ImageFont.load_default()
g = Image.new("RGB", (W * len(cols), (H + LH) * len(rows) + LH), (255, 255, 255))
d = ImageDraw.Draw(g)
for k, (lab, _) in enumerate(cols):
    d.text((k * W + 6, 3), lab, fill=(0, 0, 0), font=f)
for r, (cam, real, rl) in enumerate(rows):
    y = LH + r * (H + LH)
    d.text((6, y + 3), f"{task} / {cam}   (col 1 = {rl})", fill=(0, 0, 0), font=f)
    for k, (_, tag) in enumerate(cols):
        try:
            im = Image.open(real if tag is None else f"{look}/{tag}_{cam}.png").convert("RGB")
            if tag is None and cam == "operator":   # photo is 4:3; centre-crop to 16:9
                w0, h0 = im.size; nh = int(w0 * 9 / 16); im = im.crop((0, (h0 - nh) // 2, w0, (h0 - nh) // 2 + nh))
            g.paste(im.resize((W, H)), (k * W, y + LH))
        except Exception as e:
            d.text((k * W + 6, y + LH + 100), str(e)[:50], fill=(200, 0, 0), font=f)
g.save(out, quality=90)
print(out)
