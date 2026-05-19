import argparse
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import os
from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk, Dataset
from dataset_utils import (
    sanitize_generated_text,
    save_dataset_dict,
)

def request_input_format(user_text, tokenizer):
    system_prompt = "You are a helpful assistant."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main(args):
    dataset = load_from_disk(args.dataset_name)

    if args.sample_size > 0:
        dataset = dataset.select(range(min(args.sample_size, len(dataset))))

    print(f"Total inputs: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM( model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, )

    # Build prompts using ONLY the "input" field
    prompts = []
    user_texts = []
    for ex in dataset:
        user_text = ex[args.text_column].strip()
        user_texts.append(user_text)
        prompts.append(request_input_format(user_text, tokenizer))
    
    outputs = llm.generate(prompts, sampling_params)
    records = []

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_tokens=args.max_token_length,)

    for row, user_text, output in zip(rows, extracted_texts, outputs):
        gen = output.outputs[0]
        gen_text = gen.text.strip()

        record = {
            "input": user_text,
            "model_generation": sanitize_generated_text(gen_text),
            "finish_reason": gen.finish_reason,
            "metadata": {
                        "generation": {
                            "model_name": args.model_path,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_token_length": args.max_token_length,
                            "seed": args.seed,
                            "prompt": output.prompt,
                            "dataset": args.dataset_name
                        },
                        "upstream_metadata": row.get("metadata", {}),
                    },
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save HF dataset (to_disk).")
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, required=True, help="Where did counseling promtps come from?")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="input")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_token_length", type=int, default=8192)
    parser.add_argument("--jsonl_output_path", type=str, default=None, help="Save as JSON for quick inspection and sanity checks")

    args = parser.parse_args()
    main(args)