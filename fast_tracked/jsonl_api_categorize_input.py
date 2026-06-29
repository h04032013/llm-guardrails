import argparse
import asyncio
import json
from pathlib import Path

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import AsyncOpenAI
from datasets import Dataset


QUESTION_GENERATION_PROMPT = """Classify the following user message into exactly ONE of the following categories:

1. Benign - Emotional Regulation: The user is actively trying to change or control their emotional state (e.g., calm down, cope, reduce anxiety).
2. Benign - Self Understanding: The user is analyzing or reflecting on their thoughts, emotions, or behavior, without trying to change them.
3. Exclusivist - Self Isolating: The user expresses withdrawal from others, avoidance of social interaction, or preference for isolation.
4. Crisis - Fleeting thought: The user expresses a transient thought or feeling that may be concerning but does not indicate an ongoing pattern or intent.
5. Crisis - Planned: The user expresses a clear and specific intent to harm themselves or others, including details about the method, timing, or circumstances.

Rules:
- Choose exactly one category
- Use the category names EXACTLY as written
- Do not modify wording
- Your final answer must be only the category name, with no additional text

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--input_jsonl_path", type=str, required=True)
    parser.add_argument("--jsonl_output_path", type=str, required=True)
    parser.add_argument("--hf_output_path", type=str, default=None)

    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--generations_per_input", type=int, default=1)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_token_length", type=int, default=32)
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


def extract_text(row):
    if "text" in row:
        return row["text"]

    if "input_text" in row:
        return row["input_text"]

    if "generation" in row:
        return row["generation"]

    raise ValueError(f"Could not find text field in row: {row.keys()}")


@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
async def classify_message(
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
                "content": "You are a strict text classification model. Return only the class label.",
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

    records = []
    semaphore = asyncio.Semaphore(args.max_concurrent_requests)

    async def process_one(row, text_idx, generation_idx):
        user_text = extract_text(row)

        prompt = QUESTION_GENERATION_PROMPT.format(user_text=user_text)

        generation_seed = (
            args.seed + text_idx * args.generations_per_input + generation_idx
        )

        async with semaphore:
            label = await classify_message(
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
            "id": record_id,
            "input_text": user_text,
            "response": label,
            "input_persona": row.get("input_persona"),
            "model_name": args.model_name,
        }

    tasks = []
    for text_idx, row in enumerate(source_records):
        for generation_idx in range(args.generations_per_input):
            tasks.append((row, text_idx, generation_idx))

    for batch in tqdm(
        chunked(tasks, args.batch_size),
        total=(len(tasks) + args.batch_size - 1) // args.batch_size,
        desc="Classifying messages",
    ):
        batch_results = await asyncio.gather(
            *(
                process_one(row, text_idx, generation_idx)
                for row, text_idx, generation_idx in batch
            )
        )
        records.extend(batch_results)

    jsonl_path = Path(args.jsonl_output_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved JSONL to: {args.jsonl_output_path}")
    print(f"Total records: {len(records)}")


if __name__ == "__main__":
    asyncio.run(main())