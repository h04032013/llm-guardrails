import argparse
import json
import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from dataset_utils import (
    sanitize_generated_text,
    save_dataset_dict,
)

def request_input_format(user_prompt, tokenizer):
    system_prompt = "You are a useful assistant."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

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
    dataset = load_dataset(args.dataset_name, split=args.split)

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

    for row in dataset:
        user_text = str(row.get(args.text_column, "")).strip()
        if not user_text:
            continue

        user_prompt = build_classification_prompt(user_text)
        prompt = request_input_format(user_prompt, tokenizer)

        prompts.append(prompt)
        rows.append(row)

    print(f"Prepared {len(prompts)} prompts")
    if prompts:
        print("\nSample prompt:\n")
        print(prompts[0])

    sampling_params = SamplingParams(
        temperature=args.temperature,
        seed=args.seed,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=[args.stop] if args.stop else None,
    )

    outputs = llm.generate(prompts, sampling_params)

    records = []

    for row, output in zip(rows, outputs):
        gen = output.outputs[0]
        gen_text = gen.text.strip()

        record = {
            "input": row.get(args.text_column, ""),
            "raw_generation": gen_text,
            "predicted_category": sanitize_generated_text(gen_text),
            "finish_reason": gen.finish_reason,
            "metadata": {"dataset_name": args.dataset_name,
                        "model_name": args.model_path,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_tokens": args.max_tokens,
                        "seed": args.seed,
                        "prompt": output.prompt,
                    }
        }
        records.append(record)

    if args.json_output_path:
        os.makedirs(os.path.dirname(args.json_output_path), exist_ok=True)
        with open(args.json_output_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    print(f"Saved JSON to: {args.json_output_path}")

# Create HF dataset
    hf_dataset = Dataset.from_list(records)

# Save to disk
    hf_dataset.save_to_disk(args.output_dir)

    print(f"\nSaved Hugging Face dataset to: {args.output_dir}")
    print(hf_dataset[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify dataset inputs with a vLLM model.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_column", type=str, required=True, help="Column containing the user text to classify.")
    parser.add_argument("--sample_size", type=int, default=0, help="Number of rows to process; 0 means full split.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--json_output_path",type=str,default=None,help="Optional path to save results as a JSON file.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--stop", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)