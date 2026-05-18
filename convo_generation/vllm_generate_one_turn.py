import argparse
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset

#https://huggingface.co/datasets/proj-persona/PersonaHub 
#https://github.com/tencent-ailab/persona-hub/tree/main/code
#https://github.com/allenai/open-instruct/blob/main/scripts/persona_driven_data_gen/persona_driven_generate_math_code.py

def request_input_format(user_text, tokenizer):
    system_prompt = "You are a helpful assistant."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main(args):
    # Load dataset
    ds = load_dataset(args.dataset_name, split="train")

    if args.sample_size > 0:
        ds = ds.select(range(args.sample_size))

    print(f"Total inputs: {len(ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM( model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, max_model_len=args.max_model_len)

    # Build prompts using ONLY the "input" field
    prompts = []
    for ex in ds:
        user_text = ex["input"].strip()
        prompts.append(request_input_format(user_text, tokenizer))

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_tokens=args.max_tokens)

    outputs = llm.generate(prompts, sampling_params)

    with open(args.output_dir, "w", encoding="utf-8") as f:
        for i, out in enumerate(outputs):
            f.write(json.dumps({
                "input": ds[i]["input"],
                "generation": out.outputs[0].text,   # single output
                "finish_reason": out.outputs[0].finish_reason
            }, ensure_ascii=False) + "\n")

    print(f"Wrote: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default="ShenLab/MentalChat16K")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="input")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=2048)

    args = parser.parse_args()
    main(args)