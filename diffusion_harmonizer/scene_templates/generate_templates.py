from __future__ import annotations

import argparse

from diffusion_harmonizer.scene_templates.genie_generator_adapter import load_genie_scene_templates
from diffusion_harmonizer.scene_templates.prompt_templates import generate_templates_from_text_prompts, load_prompt_file
from diffusion_harmonizer.scene_templates.templates import generate_default_templates, save_templates, validate_templates


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or validate DiffusionHarmonizer tabletop scene templates.")
    parser.add_argument("--output", default="diffusion_harmonizer/scene_templates/default_templates.json")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--genie_templates_path", default=None, help="Optional Genie Sim generator JSON to normalize and validate.")
    parser.add_argument("--prompt_file", default=None, help="Plain-text/markdown natural-language scene prompts, one prompt per line.")
    parser.add_argument("--prompt", action="append", default=[], help="Inline natural-language scene prompt. Can be repeated.")
    args = parser.parse_args()

    if args.genie_templates_path:
        templates = load_genie_scene_templates(args.genie_templates_path)
    elif args.prompt_file or args.prompt:
        prompts = []
        if args.prompt_file:
            prompts.extend(load_prompt_file(args.prompt_file))
        prompts.extend(args.prompt)
        templates = generate_templates_from_text_prompts(prompts, count=args.count, seed=args.seed)
    else:
        templates = generate_default_templates(args.count, args.seed)
    validate_templates(templates)
    output = save_templates(templates, args.output)
    print(f"Wrote {output} with {len(templates)} validated templates")


if __name__ == "__main__":
    main()
