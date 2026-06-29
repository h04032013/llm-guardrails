import argparse
import asyncio
import json
from pathlib import Path
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import AsyncOpenAI
from datasets import load_dataset, Dataset, Features, Value, Sequence

QUESTION_GENERATION_PROMPT = (
    """Classify the following user message into exactly ONE of the following categories:

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
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="proj-persona/PersonaHub")
    parser.add_argument("--text_column", type=str, default="input")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--hf_output_path", type=str, required=True)
    parser.add_argument("--jsonl_output_path", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--generations_per_persona", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_token_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--max_concurrent_requests", type=int, default=5)
    return parser.parse_args()

@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
async def generate_conversation(client, model_name, prompt, temperature, top_p, max_token_length, seed):
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant."
                ),
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

    with open(args.api_key_path, "r") as f:
        api_key = f.read().strip()

    client = AsyncOpenAI(api_key=api_key)

    source_dataset = load_dataset(
        args.dataset_name,
        split=args.split,
    )

    if args.sample_size is not None:
        source_dataset = source_dataset.select(range(args.sample_size))

    text_column = source_dataset[args.text_column]

    records = []
    semaphore = asyncio.Semaphore(args.max_concurrent_requests)

    async def process_one(text, text_idx, generation_idx):
        prompt = QUESTION_GENERATION_PROMPT.format(user_text=text)

        generation_seed = (
            args.seed + text_idx * args.generations_per_persona + generation_idx
        )

        async with semaphore:
            content = await generate_conversation(
                client=client,
                model_name=args.model_name,
                prompt=prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_token_length=args.max_token_length,
                seed=generation_seed,
)

        record_id = text_idx * args.generations_per_persona + generation_idx

        return {
            "id": record_id,
            "input_text": text,
            "response": content,
            "source_dataset": args.dataset_name,
    }
        

    tasks = []
    for text_idx, text in enumerate(text_column):
        for generation_idx in range(args.generations_per_persona):
            tasks.append((text, text_idx, generation_idx))

    for batch in tqdm(
        chunked(tasks, args.batch_size),
        total=(len(tasks) + args.batch_size - 1) // args.batch_size,
        desc="Generating conversations",
    ):
        batch_results = await asyncio.gather(
            *(process_one(persona, persona_idx, generation_idx)
              for persona, persona_idx, generation_idx in batch)
        )
        records.extend(batch_results)

    output_dataset = Dataset.from_list(records)

    output_dataset.save_to_disk(args.hf_output_path)

    jsonl_path = Path(args.jsonl_output_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved HF dataset to: {args.hf_output_path}")
    print(f"Saved JSONL to: {args.jsonl_output_path}")
    print(f"Total records: {len(records)}")

if __name__ == "__main__":
    asyncio.run(main())