"""Run any per-task StationeryBench eval runner with a visual mode (no edits to the runner).

  python run_eval_visual.py --visual nominal|nominal2|dr_train|dr_heldout|off [--visual-level 1.0]
         [--visual-seed 0] [--visual-no-image-aug] --runner eval/run_eval_efb.py  <runner args...>

Every env created through gymnasium.make whose id starts with 'Stationery' is wrapped in
visual.VisualWrapper.  The look of episode i is a deterministic function of (visual-seed, reset
seed), so different policies are evaluated under identical looks.  Also selectable without this
script: STATIONERY_VISUAL=<mode> python <runner> ... with `import visual; visual.install_from_env()`.
"""
import os, sys  # noqa: E401  repo root on sys.path -> gyre_paths (scripts/env.sh does the same)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gyre_paths as GP  # noqa: E402
import argparse, os, runpy, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import visual  # noqa: E402

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--visual", default="nominal", choices=list(visual.MODES))
ap.add_argument("--visual-level", type=float, default=1.0)
ap.add_argument("--visual-seed", type=int, default=0)
ap.add_argument("--visual-no-image-aug", action="store_true")
ap.add_argument("--runner", required=True)
a, rest = ap.parse_known_args()
visual.install(a.visual, level=a.visual_level, vis_seed=a.visual_seed,
               image_aug=False if a.visual_no_image_aug else None)
sys.argv = [a.runner] + rest
sys.path.insert(0, os.path.dirname(os.path.abspath(a.runner)))
runpy.run_path(a.runner, run_name="__main__")
