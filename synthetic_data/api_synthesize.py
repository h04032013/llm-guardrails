import argparse
import asyncio
import json
from pathlib import Path
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import AsyncOpenAI
from datasets import load_dataset, Dataset, Features, Value, Sequence

QUESTION_GENERATION_PROMPT = (
    "Generate a user message based on the following persona as if they are venting to a chatbot: {persona}."
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--dataset_name", type=str, default="proj-persona/PersonaHub")
    parser.add_argument("--dataset_config", type=str, default="persona")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--hf_output_path", type=str, required=True)
    parser.add_argument("--jsonl_output_path", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--generations_per_persona", type=int, default=5)
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
                    "Return only the user's message, with no labels or extra commentary."
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
        args.dataset_config,
        split=args.split,
    )

    if args.sample_size is not None:
        source_dataset = source_dataset.select(range(args.sample_size))

    personas = source_dataset["persona"]

    records = []
    semaphore = asyncio.Semaphore(args.max_concurrent_requests)

    async def process_one(persona, persona_idx, generation_idx):
        prompt = QUESTION_GENERATION_PROMPT.format(persona=persona)

        generation_seed = args.seed + persona_idx * args.generations_per_persona + generation_idx

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

        record_id = persona_idx * args.generations_per_persona + generation_idx

        return {
            "id": record_id,
            "synthetic_input": content, 
            "metadata": {
                "synthesize_model_params": {
                    "seed": generation_seed,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_token_length": args.max_token_length,
                    "model_name": args.model_name,
                },
                "synthetesize_data_params": {
                    "persona_source_dataset": args.dataset_name,
                    "persona_source_dataset_config_name": args.dataset_config,
                    "persona": persona,
                    "prompt": prompt,
                    "generation_idx": generation_idx,
                },
            },
        }

    tasks = []
    for persona_idx, persona in enumerate(personas):
        for generation_idx in range(args.generations_per_persona):
            tasks.append((persona, persona_idx, generation_idx))

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