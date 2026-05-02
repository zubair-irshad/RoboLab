from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


USD_SUFFIXES = {".usd", ".usda", ".usdc"}
HDRI_SUFFIXES = {".hdr", ".exr", ".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class AssetRecord:
    path: str
    category: str
    size_bytes: int
    tags: list[str]


class AssetIndex:
    """Minimal manifest for the downloaded GenieSimAssets tree.

    GenieSimAssets is treated as the single source of truth. Category assignment
    is intentionally shallow: it uses top-level folder names and conservative
    filename heuristics rather than introducing a second registry.
    """

    def __init__(self, root: str | Path = "assets/geniesim/"):
        self.root = Path(root)
        self.records: list[AssetRecord] = []
        if self.root.exists():
            self.records = self._scan()

    def _scan(self) -> list[AssetRecord]:
        records: list[AssetRecord] = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in USD_SUFFIXES:
                category = self._category_for(path)
            elif suffix in HDRI_SUFFIXES and self._looks_like_hdri(path):
                category = "hdri"
            else:
                continue
            records.append(
                AssetRecord(
                    path=str(path),
                    category=category,
                    size_bytes=path.stat().st_size,
                    tags=self._semantic_tags(path),
                )
            )
        return records

    def _looks_like_hdri(self, path: Path) -> bool:
        rel_parts = path.relative_to(self.root).parts
        top = rel_parts[0].lower() if rel_parts else ""
        full = "/".join(part.lower() for part in rel_parts)
        return "hdri" in full or "/hdr/" in f"/{full}/" or "dome" in full or top in {"hdri", "hdr", "envmap", "environment"}

    def _category_for(self, path: Path) -> str:
        rel_parts = path.relative_to(self.root).parts
        top = rel_parts[0].lower() if rel_parts else ""
        full = "/".join(part.lower() for part in rel_parts)
        suffix = path.suffix.lower()

        if suffix in HDRI_SUFFIXES and self._looks_like_hdri(path):
            return "hdri"
        if "/light/" in f"/{full}/" or "/lights/" in f"/{full}/":
            return "light"
        if "robot" in full or top in {"robot", "robots"}:
            return "robot"
        if any(token in full for token in ("background", "scene", "room", "env")):
            return "background"
        if top in {"objects", "object", "assets", "models"}:
            return "object"
        return top or "object"

    def _semantic_tags(self, path: Path) -> list[str]:
        candidates = [
            path.with_suffix(".json"),
            path.parent / "metadata.json",
            path.parent / "semantic.json",
        ]
        tags: set[str] = set()
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text())
            except Exception:
                continue
            for key in ("tags", "semantic_tags", "categories", "labels"):
                value = payload.get(key) if isinstance(payload, dict) else None
                if isinstance(value, str):
                    tags.add(value)
                elif isinstance(value, Iterable):
                    tags.update(str(item) for item in value)
        return sorted(tags)

    def write_manifest(self, output_path: str | Path = "assets/geniesim/manifest.json") -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": str(self.root),
            "count": len(self.records),
            "assets": [asdict(record) for record in self.records],
        }
        output.write_text(json.dumps(payload, indent=2))
        return output

    def _paths(self, category: str) -> list[Path]:
        return [Path(record.path) for record in self.records if record.category == category]

    def objects(self, category: str | None = None) -> list[Path]:
        paths = [path for path in self._paths("object") if path.suffix.lower() in USD_SUFFIXES]
        if category:
            needle = category.lower()
            paths = [path for path in paths if needle in str(path).lower()]
        return paths

    def hdris(self) -> list[Path]:
        return self._paths("hdri")

    def robots(self) -> list[Path]:
        return self._paths("robot")

    def backgrounds(self) -> list[Path]:
        return self._paths("background")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="assets/geniesim/")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    index = AssetIndex(args.root)
    output = args.output or str(Path(args.root) / "manifest.json")
    manifest = index.write_manifest(output)
    print(f"Wrote {manifest} with {len(index.records)} assets")


if __name__ == "__main__":
    main()
