"""Post-hoc paired-data builders - run after capture_harmonizer_views.py.

Reads the per-view captures under ``data/captures/<env>/sphere/<NNNN>/`` and
emits paired training data under ``data/diffusion_harmonizer/<env>/``. Does
**not** import Isaac Sim — pure NumPy / OpenCV (and torch + gsplat for the
artifacts and reinsertion builders, which are added in a follow-up).

Usage::

    PYTHONPATH=. python scripts/build_harmonizer_pairs.py \
        --captures data/captures \
        --output-root data/diffusion_harmonizer \
        --components isp shadow
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", default="data/captures")
    parser.add_argument("--output-root", default="data/diffusion_harmonizer")
    parser.add_argument("--env", nargs="*", default=None,
                        help="Restrict to specific env names; default = all envs found under --captures.")
    parser.add_argument("--components", nargs="+", default=["isp", "shadow"],
                        choices=("isp", "shadow"),
                        help="Builders to run. (artifacts and reinsertion will follow once their post-hoc forms land.)")
    parser.add_argument("--isp-pairs", type=int, default=12)
    parser.add_argument("--shadow-pairs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _enumerate_envs(captures_root: Path) -> list[Path]:
    return sorted(p for p in captures_root.iterdir() if p.is_dir() and (p / "manifest.json").exists())


def main() -> None:
    args = parse_args()
    captures_root = Path(args.captures)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    env_dirs = _enumerate_envs(captures_root)
    if args.env:
        wanted = set(args.env)
        env_dirs = [p for p in env_dirs if p.name in wanted]
    if not env_dirs:
        raise SystemExit(f"No env captures found under {captures_root}. Run capture_harmonizer_views.py first.")

    # Defer imports so the script is fast to start when only one builder is selected.
    from diffusion_harmonizer.builders import isp as isp_builder, shadow as shadow_builder

    summary: dict = {"envs": {}}
    for env_dir in env_dirs:
        env_name = env_dir.name
        env_summary: dict = {"components": {}}
        try:
            if "isp" in args.components:
                env_summary["components"]["isp_modification"] = {
                    "num_pairs": len(isp_builder.build(
                        env_dir,
                        output_root / env_name / "02_isp_modification",
                        count=args.isp_pairs,
                        seed=args.seed,
                    )),
                }
            if "shadow" in args.components:
                env_summary["components"]["shadow_simulation"] = {
                    "num_pairs": len(shadow_builder.build(
                        env_dir,
                        output_root / env_name / "04_shadow_simulation",
                        count=args.shadow_pairs,
                    )),
                }
        except Exception as exc:
            env_summary["error"] = str(exc)
            env_summary["traceback"] = traceback.format_exc()
            print(f"[builder:{env_name}] FAILED: {exc}", flush=True)
            traceback.print_exc()
        summary["envs"][env_name] = env_summary
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[builder] wrote summary {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
