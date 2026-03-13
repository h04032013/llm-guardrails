#!/bin/bash
#SBATCH --job-name=B_prmpt
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --mem=16G
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
out_path=/n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails/files/clean_og_120b.jsonl
in_path=/n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails/files/100_ogprmpt_120b_prompts.jsonl
tensor_parallel_size=1

PYTHONPATH=. python synthetic_data/output_parser.py --in_path $in_path --out_path $out_path