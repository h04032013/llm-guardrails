#!/bin/bash
#SBATCH --job-name=oss1x1_q
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:10:00
#SBATCH --mem=32G
#SBATCH --output=files/log_files/ossoutput.out
#SBATCH --error=files/log_files/osserror.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu

cd /n/holylabs/LABS/dam_lab/Users/hdiaz/synthetic_data_persona

export HF_HOME="/n/netscratch/dam_lab/Lab/hdiaz/hgf_new_hub"
module purge
module load Mambaforge
module load cuda cudnn
mamba activate ossenv

sample_size=10  # Set sample_size=0 if you want to use the full version of 200k psersonas.
out_path=/n/holylabs/LABS/dam_lab/Users/hdiaz/synthetic_data_persona/files/outputs/ossfr_1gpu_1pp.jsonl
#model_path=Qwen/Qwen3-8B
num_questions_per_persona=1
#model_path=allenai/Olmo-3-7B-Think
model_path=openai/gpt-oss-20b
tensor_parallel_size=1

PYTHONPATH=. python generate/vllm_synthesize.py --model_path $model_path --sample_size $sample_size  --output_path $out_path --num_questions_per_persona $num_questions_per_persona  --tensor_parallel_size $tensor_parallel_size
