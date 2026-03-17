# /n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/input_data
import argparse
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def request_input_format(user_prompt, tokenizer):
    system_prompt = (
        "You are a useful assistant."
    )
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

1. Benign - Emotional Regulation: the user is trying to manage or regulate emotions (e.g., calming down, coping with stress, handling feelings).
2. Benign - Self Understanding: the user is trying to understand their thoughts, feelings, or behavior.
3. Exclusivist - Self Isolating: the user expresses social withdrawal, isolation, or distancing from others.

Rules:
- Choose exactly one category
- Use the category names EXACTLY as written
- Do not modify wording
- Output in this format:

Category: <category>
User message:
\"\"\"{user_text}\"\"\" 
"""
    
def main(args):
    dataset = load_dataset(args.dataset_name, split=args.split)

    if args.sample_size > 0:
        sample_size = min(args.sample_size, len(dataset))
        dataset = dataset.select(range(sample_size))

    print(f"Loaded split '{args.split}' with {len(dataset)} rows")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    prompts = []
    rows = []

    for row in dataset:
        user_text = (row.get(args.text_column) or "").strip()

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

    stop_list = [args.stop] if args.stop else None

    sampling_params = SamplingParams(
        temperature=args.temperature,
        seed=args.seed,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=stop_list,
    )

    outputs = llm.generate(prompts, sampling_params)

    with open(args.output_path, "w", encoding="utf-8") as out:
        for i, output in enumerate(outputs):
            original_row = rows[i]

            record = {
                "input": original_row.get("input", ""),
                "reference_output": original_row.get("output", ""),
                "prompt": output.prompt,
                "generations": [
                    {"text": o.text,
                        "finish_reason": o.finish_reason,}
                    for o in output.outputs
                ],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved results to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify MentalChat16K inputs with a vLLM model.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Hugging Face dataset name.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use.")
    parser.add_argument("--text_column", type=str, required=True, help="Column containing the user text to classify.")
    parser.add_argument("--sample_size", type=int, required=True, help="Number of rows to process; use 0 for full split.")
    parser.add_argument("--model_path", type=str,required=True,help="Path or HF name of the model.")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--temperature",type=float, default=0.0, help="Use low temperature for stable classification.")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p for sampling.")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max generation length.")
    parser.add_argument("--stop", type=str, default=None, help="Optional stop string.")
    parser.add_argument("--seed", type=int,default=0,help="Random seed.")

    args = parser.parse_args()
    main(args)