"""Merge a LoRA adapter into its base model (deployment step M4)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOGGER = logging.getLogger("atom_slm.merge")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--base", required=True, help="Base model id or path.")
    parser.add_argument("--adapter", required=True, help="Adapter checkpoint id or path.")
    parser.add_argument("--output", required=True, help="Directory for the merged model.")
    parser.add_argument("--push-to-hub", default=None, help="Optional Hub repo id.")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args(argv)


def merge(args: argparse.Namespace) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .utils import human_bytes

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    adapter_config = Path(args.adapter) / "adapter_config.json"
    if not adapter_config.exists():
        LOGGER.info("No adapter_config.json found: %s looks like a full fine-tune.", args.adapter)
        model = AutoModelForCausalLM.from_pretrained(args.adapter, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    else:
        from peft import PeftModel

        LOGGER.info("Loading base model %s ...", args.base)
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16)
        LOGGER.info("Applying adapter %s ...", args.adapter)
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(args.base)

    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    size = sum(p.stat().st_size for p in output.glob("*.safetensors"))
    LOGGER.info("Merged model saved to %s (%s)", output, human_bytes(size))

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(output), repo_id=args.push_to_hub)
        LOGGER.info("Pushed to https://huggingface.co/%s", args.push_to_hub)

    return output


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")
    merge(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
