import argparse
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset

#https://huggingface.co/datasets/proj-persona/PersonaHub 
#https://github.com/tencent-ailab/persona-hub/tree/main/code
#https://github.com/allenai/open-instruct/blob/main/scripts/persona_driven_data_gen/persona_driven_generate_math_code.py

def request_input_format(user_prompt, tokenizer):
    system_prompt = "You are a helpful assistant."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False) #removed thinking to prototype qwen fast
    
    return text 

def main(args):
    math_template = """Generate a user conversation with following persona as if they are venting to a chatbot: {persona}."""

    #Load the dataset 
    persona_dataset = load_dataset("proj-persona/PersonaHub", data_files="persona.jsonl")['train']
    if args.sample_size > 0:
        persona_dataset = persona_dataset[:args.sample_size]
    print(f"Total number of input personas: {len(persona_dataset['persona'])}")

    # Load the model and tokenizer
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=args.tensor_parallel_size) # please set tensor_parallel_size based on the GPUs you are using

    prompts = []
    max_tokens_questions = args.max_tokens_questions

    for persona in persona_dataset['persona']:
        persona = persona.strip()
        user_prompt = math_template.format(persona=persona)
        prompt = request_input_format(user_prompt, tokenizer)
        prompts.append(prompt)

    print(f"Loaded {len(prompts)} entries to process...\n\n")
    print(f"Sample 0: {prompts[0]}")

    #Personahub defaults were temperature=0.7, seed=0, top_p=0.95, max_tokens=2048
    sampling_params = SamplingParams(temperature=args.temperature, n=args.num_questions_per_persona, seed =args.seed, top_p=args.top_p, max_tokens=args.max_tokens_questions, stop = [args.stop] if args.stop else [tokenizer.eos_token])
    outputs = llm.generate(prompts, sampling_params)

    with open(args.output_path, 'w') as out:
        for i, output in enumerate(outputs):
            data = {
                "prompt": output.prompt,
                "input_persona": persona_dataset["persona"][i].strip(),
                "generations": [
                    {"text": o.text, "finish_reason": o.finish_reason}
                    for o in output.outputs
                ],
            }
            out.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"Outputted the results to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize text using a specified model and template.")
    parser.add_argument('--sample_size', required=True, type=int, help='Number of samples to process from the dataset; Set it to 0 if you want to use the full set of 200k personas.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file.')
    parser.add_argument('--num_questions_per_persona', type=int, default=1, help='Number of generations per persona.')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Number of GPUs for tensor parallelism.')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_tokens_questions', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0.95, help="top_p")
    parser.add_argument('--stop', type=str, default=None, help='Optional extra stop token.')
    args = parser.parse_args()
    main(args) 