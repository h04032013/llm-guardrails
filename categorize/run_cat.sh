#!/bin/bash
#SBATCH --job-name=16k_cat_vllm
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=0-00:30:00
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

sample_size=50  # Set sample_size=0 if you want to use the full version of 200k psersonas.
dataset_name="ShenLab/MentalChat16K"
text_column="input"
model_path="openai/gpt-oss-120b"
output_path="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/input_16k_categorized_vllm_512.jsonl"

tensor_parallel_size=2

PYTHONPATH=. python categorize/categorize_input_16k.py --dataset_name $dataset_name --text_column $text_column --sample_size $sample_size --model_path $model_path --output_path $output_path --tensor_parallel_size $tensor_parallel_size