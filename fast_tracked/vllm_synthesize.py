import argparse
from transformers import AutoTokenizer
import json
from vllm import LLM, SamplingParams
from datasets import load_dataset
from dataset_utils import (
    sanitize_generated_text,
    save_dataset_dict,)


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
        enable_thinking=False,
    )

def main(args):
    template = (
        "Generate a user message based on the following persona as if they are venting to a chatbot: {persona}."
    )

    persona_dataset = load_dataset(
        "proj-persona/PersonaHub",
        data_files="persona.jsonl",
    )["train"]

    if args.sample_size > 0:
        persona_dataset = persona_dataset.select(range(args.sample_size))

    print(f"Total number of input personas: {len(persona_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=4096,
    )

    prompts = []

    for persona in persona_dataset["persona"]:
        persona = persona.strip()
        user_prompt = template.format(persona=persona)
        prompt = request_input_format(user_prompt, tokenizer)
        prompts.append(prompt)

    print(f"Loaded {len(prompts)} entries to process...\n")
    print(f"Sample 0: {prompts[0]}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        n=args.num_questions_per_persona,
        seed=args.seed,
        top_p=args.top_p,
        max_tokens=args.max_tokens_questions,
        stop=[args.stop] if args.stop else None,
    )

    outputs = llm.generate(prompts, sampling_params)

    with open(args.output_path, "w") as out:
        for i, output in enumerate(outputs):
            data = {
                "input_persona": persona_dataset["persona"][i].strip(),
                "generations": [
                    {"text": sanitize_generated_text(o.text.strip()), "finish_reason": o.finish_reason}
                    for o in output.outputs
                ],
            }
            out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Outputted the results to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize text using a specified model and template."
    )
    parser.add_argument("--sample_size", required=True, type=int)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens_questions", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--num_questions_per_persona", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--stop", type=str, default=None)

    args = parser.parse_args()
    main(args)