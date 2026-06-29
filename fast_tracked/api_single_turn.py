import argparse
import asyncio
import json
from pathlib import Path

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import AsyncOpenAI

PROMPT_TEMPLATE = """{user_text}"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--api_key_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--input_jsonl_path", type=str, required=True)
    parser.add_argument("--jsonl_output_path", type=str, required=True)

    parser.add_argument("--text_column", type=str, default="input_text")
    parser.add_argument("--label_column", type=str, default="response")

    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--generations_per_input", type=int, default=1)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_token_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--max_concurrent_requests", type=int, default=5)

    return parser.parse_args()


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
async def generate_response(
    client,
    model_name,
    prompt,
    temperature,
    top_p,
    max_token_length,
    seed,
):
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_token_length,
        seed=seed,
    )

    return response.choices[0].message.content.strip()


def chunked(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]


async def main():
    args = parse_args()

    with open(args.api_key_path, "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    client = AsyncOpenAI(api_key=api_key)

    source_records = load_jsonl(args.input_jsonl_path)

    if args.sample_size is not None:
        source_records = source_records[:args.sample_size]

    semaphore = asyncio.Semaphore(args.max_concurrent_requests)

    async def process_one(record, text_idx, generation_idx):
        user_text = record[args.text_column]
        original_label = record.get(args.label_column)

        prompt = PROMPT_TEMPLATE.format(user_text=user_text)

        generation_seed = (
            args.seed + text_idx * args.generations_per_input + generation_idx
        )

        async with semaphore:
            model_response = await generate_response(
                client=client,
                model_name=args.model_name,
                prompt=prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_token_length=args.max_token_length,
                seed=generation_seed,
            )

        record_id = text_idx * args.generations_per_input + generation_idx

        return {
            "original_id": record.get("id"),
            "input_text": user_text,
            "original_label": original_label,
            "model_response": model_response,
            "model_name": args.model_name,
        }

    tasks = []
    for text_idx, record in enumerate(source_records):
        for generation_idx in range(args.generations_per_input):
            tasks.append((record, text_idx, generation_idx))

    output_records = []

    for batch in tqdm(
        chunked(tasks, args.batch_size),
        total=(len(tasks) + args.batch_size - 1) // args.batch_size,
        desc="Generating responses",
    ):
        batch_results = await asyncio.gather(
            *(
                process_one(record, text_idx, generation_idx)
                for record, text_idx, generation_idx in batch
            )
        )
        output_records.extend(batch_results)

    jsonl_path = Path(args.jsonl_output_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved JSONL to: {args.jsonl_output_path}")
    print(f"Total records: {len(output_records)}")

if __name__ == "__main__":
    asyncio.run(main())