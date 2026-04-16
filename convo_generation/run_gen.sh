
cd /n/llm-guardrails

export HF_HOME="/n/hgf_new_hub"
module purge
module load Mambaforge
module load cuda cudnn
mamba activate ossenv

sample_size=0  # Set sample_size=0 if you want to use the full version of 200k psersonas.
out_path=/llm-guardrails/files/shen_lab_120b.jsonl
model_path=openai/gpt-oss-120b
tensor_parallel_size=2

PYTHONPATH=. python convo_generation/vllm_generate.py --model_path $model_path --sample_size $sample_size  --output_path $out_path  --tensor_parallel_size $tensor_parallel_size
