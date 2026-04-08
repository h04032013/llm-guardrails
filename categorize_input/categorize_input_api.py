import argparse
import asyncio
import json
import random
from typing import Any, Dict, List
from datasets import load_dataset
from openai import AsyncOpenAI

PROMPT_TEMPLATE = """Classify the following user message into exactly ONE of the following categories:

1. Benign - Emotional Regulation: the user is trying to manage or regulate emotions (e.g., calming down, coping with stress, handling feelings).
2. Benign - Self Understanding: the user is trying to understand their thoughts, feelings, or behavior.
3. Exclusivist - Self Isolating: the user expresses social withdrawal, isolation, or distancing from others.

Rules:
- Choose exactly one category
- Use the category names EXACTLY as written
- Do not modify wording
- Your final answer must be only the category name, with no additional text (e.g., 'Benign - Emotional Regulation')
User message: {user_message}
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample_size", type=int, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_concurrent_requests", type=int, default=10)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--input_column", type=str, default="input")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_output_tokens", type=int, default=256)
    return parser.parse_args()

def parse_last_line(text: str) -> str:
    return text.strip().split("\n")[-1].strip()

async def generate_one(client, semaphore, row_idx, row, model_name, input_column, temperature, max_output_tokens):
    user_message = row.get(input_column, "")
    if not isinstance(user_message, str):
        user_message = str(user_message)

    prompt = PROMPT_TEMPLATE.format(user_message=user_message)

    async with semaphore:
        try:
            response = await client.responses.create(
                model=model_name,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

            full_generation = response.output_text.strip()
            parsed = parse_last_line(full_generation)

            return {
                "row_index": row_idx,
                "user_message": user_message,
                "full_generation": full_generation,
                "parsed": parsed,
                "raw_row": row,
            }
        except Exception as e:
            return {
                "row_index": row_idx,
                "user_message": user_message,
                "full_generation": None,
                "parsed": None,
                "raw_row": row,
                "error": str(e),
            }

async def main():
    args = parse_args()

    client = AsyncOpenAI(api_key=args.api_key)
    dataset = load_dataset(args.dataset_name, split=args.split)

    n = min(args.sample_size, len(dataset))
    indices = random.Random(args.seed).sample(range(len(dataset)), n)
    rows = [dataset[i] for i in indices]

    semaphore = asyncio.Semaphore(args.max_concurrent_requests)

    tasks = [
        generate_one(client, semaphore, idx, row,
                     args.model_name, args.input_column,
                     args.temperature, args.max_output_tokens)
        for idx, row in zip(indices, rows)
    ]

    results = await asyncio.gather(*tasks)

    with open(args.output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(results)} results to {args.output_path}")

if __name__ == "__main__":
    asyncio.run(main())