#!/bin/bash
#SBATCH --job-name=B_prmpt
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=0-03:00:00
#SBATCH --mem=128G
#SBATCH --output=files/log_files/output.out
#SBATCH --error=files/log_files/error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu

cd /n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails

export HF_HOME="/n/netscratch/dam_lab/Lab/hdiaz/hgf_new_hub"
module purge
module load Mambaforge
module load cuda cudnn
mamba activate ossenv

sample_size=100  # Set sample_size=0 if you want to use the full version of 200k psersonas.
out_path=/n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails/files/zero_shot_100_ogprmpt_120b_prompts.jsonl
#model_path=Qwen/Qwen3-8B
#model_path=allenai/Olmo-3-7B-Think
model_path=openai/gpt-oss-120b
tensor_parallel_size=2

PYTHONPATH=. python synthetic_data/vllm_synthesize.py --model_path $model_path --sample_size $sample_size  --output_path $out_path  --tensor_parallel_size $tensor_parallel_size
