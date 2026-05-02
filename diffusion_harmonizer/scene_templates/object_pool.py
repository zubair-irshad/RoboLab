from __future__ import annotations

import random
from pathlib import Path

from diffusion_harmonizer.asset_manager import AssetIndex
from diffusion_harmonizer.scene_templates.templates import SceneTemplate


class ObjectPoolSampler:
    """Samples USD foreground objects per scene template.

    Assets are selected per generated scene/pair, not once for the whole run.
    This avoids the repeated-bottle failure mode while keeping object choices
    tied to the template's natural-language domain.
    """

    def __init__(self, index: AssetIndex, seed: int = 42):
        self.index = index
        self.rng = random.Random(seed)
        self._textured_dirs = self._find_textured_asset_dirs(index)
        self._all_objects = [path for path in index.objects() if self._is_scene_asset(path)]
        self._textured_objects = [path for path in self._all_objects if self._has_nearby_textures(path)]
        self._all_hdris = list(index.hdris())

    def sample(self, template: SceneTemplate, count: int | None = None) -> list[Path]:
        if not self._all_objects:
            return []
        lo, hi = template.object_count
        requested = count if count is not None else self.rng.randint(lo, hi)
        candidates = self._ranked_candidates(template, prefer_textured=True)
        if len(candidates) < requested:
            fallback = self._ranked_candidates(template, prefer_textured=False)
            candidates = candidates + [path for path in fallback if path not in candidates]
        if len(candidates) <= requested:
            return candidates[:requested]
        head = candidates[: max(requested * 4, requested)]
        return self.rng.sample(head, requested)

    def sample_hdri(self, template: SceneTemplate) -> Path | None:
        if not self._all_hdris:
            return None
        query_tokens = [token.strip().lower() for token in (template.hdri_query + "," + template.background_query).replace(" ", ",").split(",") if token.strip()]
        wants_indoor = any(token in query_tokens for token in ("indoor", "home", "room", "kitchen", "office", "supermarket", "laboratory", "cafe", "studio"))
        ranked = []
        for path in self._all_hdris:
            low = str(path).lower()
            score = sum(token in low for token in query_tokens)
            if wants_indoor and any(token in low for token in ("home_", "laboratory", "supermarket", "kitchen", "office", "studio")):
                score += 4
            if wants_indoor and any(token in low for token in ("airport", "field", "grass", "road", "street", "desert", "sky", "outdoor", "hdr-skies")):
                score -= 3
            ranked.append((score, self.rng.random(), path))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        head = ranked[: min(4, len(ranked))]
        return self.rng.choice(head)[2]

    def _ranked_candidates(self, template: SceneTemplate, prefer_textured: bool) -> list[Path]:
        pool = self._textured_objects if prefer_textured and self._textured_objects else self._all_objects
        ranked = []
        for path in pool:
            low = str(path).lower()
            score = sum(query.lower() in low for query in template.object_queries)
            if prefer_textured:
                score += 2
            if any(token in low for token in ("building_block", "building_blocks", "cube", "cuboid", "cylinder")):
                score -= 2
            ranked.append((score, self.rng.random(), path))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in ranked if item[0] > 0] or [item[2] for item in ranked]

    @staticmethod
    def _is_scene_asset(path: Path) -> bool:
        if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            return False
        low = "/" + str(path).lower().replace("\\", "/")
        if any(token in low for token in ("/texture/", "/textures/", "/material/", "/materials/")):
            return False
        return True

    def _has_nearby_textures(self, path: Path) -> bool:
        return path.parent in self._textured_dirs

    @staticmethod
    def _find_textured_asset_dirs(index: AssetIndex) -> set[Path]:
        image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".hdr", ".bmp", ".tga"}
        textured_dirs: set[Path] = set()
        for record in index.records:
            path = Path(record.path)
            if path.suffix.lower() not in image_suffixes:
                continue
            textured_dirs.add(path.parent)
            if path.parent.name.lower() in {"texture", "textures", "material", "materials"}:
                textured_dirs.add(path.parent.parent)
        return textured_dirs
