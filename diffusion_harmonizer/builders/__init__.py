"""Post-hoc paired-data builders that read disk captures from ``capture.py``.

These do **not** import Isaac Sim. They consume the per-view directories
produced by ``diffusion_harmonizer.capture.capture_envs`` and emit paired
training data on disk. Each builder is independent and can be run / re-run
without touching the live simulator.
"""

from . import artifacts, isp, shadow

__all__ = ["artifacts", "isp", "shadow"]
