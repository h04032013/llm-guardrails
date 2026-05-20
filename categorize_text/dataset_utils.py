from __future__ import annotations

import re
from typing import Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_GPT_OSS_FINAL_TOKEN = "assistantfinal"
_LEADING_CONTROL_TOKEN_RE = re.compile(
    r"^\s*(?:"
    r"<\|assistant\|>"
    r"|<\|user\|>"
    r"|<\|system\|>"
    r"|<\|im_start\|>\s*(?:assistant|user|system)?"
    r"|<\|im_end\|>"
    r"|<\|endoftext\|>"
    r"|<\|begin_of_text\|>"
    r"|<\|eot_id\|>"
    r"|<\|start_header_id\|>assistant<\|end_header_id\|>"
    r"|<s>"
    r"|</s>"
    r"|<bos>"
    r"|</bos>"
    r"|<eos>"
    r"|</eos>"
    r")\s*",
    flags=re.IGNORECASE,
)
_TRAILING_CONTROL_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"<\|assistant\|>"
    r"|<\|user\|>"
    r"|<\|system\|>"
    r"|<\|im_start\|>\s*(?:assistant|user|system)?"
    r"|<\|im_end\|>"
    r"|<\|endoftext\|>"
    r"|<\|begin_of_text\|>"
    r"|<\|eot_id\|>"
    r"|<\|start_header_id\|>assistant<\|end_header_id\|>"
    r"|<s>"
    r"|</s>"
    r"|<bos>"
    r"|</bos>"
    r"|<eos>"
    r"|</eos>"
    r")\s*$",
    flags=re.IGNORECASE,
)


def load_persona_dataset(
    dataset_name: str,
    dataset_config_name: Optional[str] = "persona",
    seed: int = 42,
    sample_size: int = 0,
) -> Dataset:
    if dataset_config_name:
        dataset = load_dataset(dataset_name, dataset_config_name)["train"]
    else:
        dataset = load_dataset(dataset_name)["train"]

    persona_idx = list(range(len(dataset)))
    dataset = dataset.add_column("persona_idx", persona_idx)
    dataset = dataset.shuffle(seed=seed)

    if sample_size > 0:
        dataset = dataset.select(range(min(sample_size, len(dataset))))

    return dataset


def save_dataset_dict(dataset: Dataset, output_dir: str) -> None:
    DatasetDict({"train": dataset}).save_to_disk(output_dir)


def load_saved_dataset(input_dir: str) -> Dataset:
    dataset = load_from_disk(input_dir)
    if isinstance(dataset, DatasetDict):
        return dataset["train"]
    return dataset


def sanitize_generated_text(text: str) -> str:
    text = text or ""

    idx = text.rfind(_GPT_OSS_FINAL_TOKEN)
    if idx != -1:
        text = text[idx + len(_GPT_OSS_FINAL_TOKEN) :]

    cleaned = _THINK_BLOCK_RE.sub("", text)

    while True:
        updated = _LEADING_CONTROL_TOKEN_RE.sub("", cleaned)
        updated = _TRAILING_CONTROL_TOKEN_RE.sub("", updated)
        if updated == cleaned:
            break
        cleaned = updated

    return cleaned.strip()


def trim_after_anchor(text: str, anchor: str | None) -> str:
    if not anchor:
        return text

    idx = text.rfind(anchor)
    if idx == -1:
        return text.strip()

    return text[idx + len(anchor) :].strip()


def clean_generated_text(text: str, anchor: str | None = None) -> str:
    return trim_after_anchor(sanitize_generated_text(text), anchor)