#!/bin/bash
#SBATCH --job-name=one_turn_shen
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=0-00:15:00
#SBATCH --mem=128G
#SBATCH --output=files/logs/%x_%j.out
#SBATCH --error=files/logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu 

set -euo pipefail

env_directory_setup() {
    module purge
    module load python
    mamba activate ossenv
    cd /n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails
}

make_directory() {
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "files/logs"
}

env_directory_setup
export HF_HOME="/n/netscratch/dam_lab/Lab/hdiaz/hgf_new_hub"

SAMPLE_SIZE=50
DATASET_NAME="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/categorized_input/HF_FORMAT/ShenLab_MentalChat16K/openai_gpt-oss-20b/0samples"
#DATASET_NAME="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/categorized_input/HF_FORMAT/synthetic_data/openai_gpt-oss-20b/0samples"
MODEL_PATH="openai/gpt-oss-120b"
SPLIT="train"
MODEL_SLUG="${MODEL_PATH//\//_}"
TEXT_COLUMN="input"
OUTPUT_DIR="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/generated_responses/one_turn/HF_FORMAT/MentalChat16k(fr)/${MODEL_SLUG}/${SAMPLE_SIZE}samples"
TEMPERATURE=0.7
TOP_P=1.0
MAX_TOKEN_LENGTH=8192
SEED=42
TENSOR_PARALLEL_SIZE=2
JSONL_OUTPUT_PATH="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/generated_responses/one_turn/JSONL_FORMAT/MentalChat16k/${MODEL_SLUG}/${SAMPLE_SIZE}samples/${SPLIT}_${TEXT_COLUMN}.jsonl"

make_directory

echo "========================================"
echo "Job ID:              $SLURM_JOB_ID"
echo "Node:                $SLURMD_NODENAME"
echo "Model:               $MODEL_PATH"
echo "Dataset:             $DATASET_NAME"
echo "Split:               $SPLIT"
echo "Text column:         $TEXT_COLUMN"
echo "Sample size:         $SAMPLE_SIZE"
echo "Seed:                $SEED"
echo "Temperature:         $TEMPERATURE"
echo "Top-p:               $TOP_P"
echo "Max token length:    $MAX_TOKEN_LENGTH"
echo "Output:              $OUTPUT_DIR"
echo "Start time:          $(date)"
echo "HF_HOME:             $HF_HOME"
echo "========================================"

python convo_generation/vllm_generate_one_turn.py \
    --split "$SPLIT" \
    --text_column "$TEXT_COLUMN" \
    --model_path "$MODEL_PATH" \
    --sample_size $SAMPLE_SIZE \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --temperature $TEMPERATURE \
    --jsonl_output_path "$JSONL_OUTPUT_PATH" \
    --max_token_length $MAX_TOKEN_LENGTH \
    --top_p $TOP_P \
    --seed $SEED
