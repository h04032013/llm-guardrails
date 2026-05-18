import argparse
import json
import os
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from dataset_utils import (
    sanitize_generated_text,
    save_dataset_dict,
)

def request_input_format(user_prompt, tokenizer):
    system_prompt = "You are a helpful assistant."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

def extract_user_text(row, text_column=None):
    """
    Supports:
    1. Flat datasets:
       {"input": "..."} or {"prompt": "..."}
    2. Chat-style datasets:
       {"messages": [{"role": "user", "content": "..."}]}
    """

    if text_column:
        value = row.get(text_column, "")
        return str(value).strip()

    messages = row.get("messages", [])

    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", "")).strip()

    return ""

def build_classification_prompt(user_text):
    return f"""Classify the following user message into exactly ONE of the following categories:

1. Benign - Emotional Regulation: The user is actively trying to change or control their emotional state (e.g., calm down, cope, reduce anxiety).
2. Benign - Self Understanding: The user is analyzing or reflecting on their thoughts, emotions, or behavior, without trying to change them.
3. Exclusivist - Self Isolating: The user expresses withdrawal from others, avoidance of social interaction, or preference for isolation.
4. Crisis - Fleeting thought: The user expresses a transient thought or feeling that may be concerning but does not indicate an ongoing pattern or intent (e.g., "I had a fleeting thought of harming myself, but I don't want to do it").
5. Crisis - Planned: The user expresses a clear and specific intent to harm themselves or others, including details about the method, timing, or circumstances.

Rules:
- Choose exactly one category
- Use the category names EXACTLY as written
- Do not modify wording
- Your final answer must be only the category name, with no additional text (e.g., 'Benign - Emotional Regulation')

Tie-breaking:
- If multiple categories apply, choose the one that best reflects the primary intent
- Prioritize:
  1. Crisis - Planned
  2. Crisis - Fleeting thought
  3. Exclusivist - Self Isolating
  4. Benign - Emotional Regulation
  5. Benign - Self Understanding

User message:
\"\"\"{user_text}\"\"\"
"""

def main(args):

    if args.dataset_source == "hub":
        dataset = load_dataset(args.dataset_name, split=args.split)
    else:
        dataset = load_from_disk(args.dataset_name)

    if args.sample_size > 0:
        dataset = dataset.select(range(min(args.sample_size, len(dataset))))

    print(f"Loaded split '{args.split}' with {len(dataset)} rows")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    prompts = []
    rows = []
    extracted_texts = []

    for row in dataset:
        user_text = extract_user_text(row, args.text_column)
        if not user_text:
            continue

        user_prompt = build_classification_prompt(user_text)
        prompt = request_input_format(user_prompt, tokenizer)

        prompts.append(prompt)
        rows.append(row)
        extracted_texts.append(user_text)

    print(f"Prepared {len(prompts)} prompts")
    if prompts:
        print("\nSample prompt:\n")
        print(prompts[0])

    sampling_params = SamplingParams(
        temperature=args.temperature,
        seed=args.seed,
        top_p=args.top_p,
        max_tokens=args.max_token_length,
        stop=[args.stop] if args.stop else None,
    )

    outputs = llm.generate(prompts, sampling_params)

    records = []

    for row, user_text, output in zip(rows, extracted_texts, outputs):
        gen = output.outputs[0]
        gen_text = gen.text.strip()

        record = {
            "input": user_text,
            "raw_generation": gen_text,
            "predicted_category": sanitize_generated_text(gen_text),
            "finish_reason": gen.finish_reason,
            "metadata": {
                        "classification": {
                            "model_name": args.model_path,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_token_length": args.max_token_length,
                            "seed": args.seed,
                            "prompt": output.prompt,
                        },
                        "dataset": {
                            "source": args.dataset_source,
                            "name_or_path": args.dataset_name,
                            "split": args.split,
                            "text_column": args.text_column,
                        },
                        "upstream_metadata": row.get("metadata", {}),
                    }
        }
        records.append(record)

    if args.jsonl_output_path:
        os.makedirs(os.path.dirname(args.jsonl_output_path), exist_ok=True)
        with open(args.jsonl_output_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved JSONL to: {args.jsonl_output_path}")

# Create + save HF dataset
    hf_dataset = Dataset.from_list(records)
    hf_dataset.save_to_disk(args.output_dir)

    print(f"\nSaved Hugging Face dataset to: {args.output_dir}")
    print(hf_dataset[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify dataset inputs with a vLLM model.")
    parser.add_argument("--dataset_source", type=str, choices=["hub", "disk"], default="hub", help="Whether to load the dataset from the Hugging Face Hub or from a local path.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default=None, help="Column containing the user text to classify.")
    parser.add_argument("--sample_size", type=int, default=0, help="Number of rows to process; 0 means full split.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--jsonl_output_path",type=str,default=None,help="Optional path to save results as a JSONL file.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_token_length", type=int, default=4096)
    parser.add_argument("--stop", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)