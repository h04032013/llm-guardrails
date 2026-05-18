#!/bin/bash
#SBATCH --job-name=api_syn_1000x5
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=0-01:00:00
#SBATCH --mem=32G
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

SAMPLE_SIZE=1000
NUM_GENERATIONS_PER_PERSONA=5
SEED=42
TEMPERATURE=1.0
TOP_P=1.0
MAX_TOKEN_LENGTH=512
DATASET_NAME="proj-persona/PersonaHub"
DATASET_CONFIG_NAME="persona"

DATASET_SLUG="${DATASET_NAME//\//_}"
MODEL_NAME="gpt-4.1-mini"
MODEL_SLUG="${MODEL_NAME//\//_}"
SPLIT="train"
API_KEY_PATH="/n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails/files/api_key.txt"
OUTPUT_DIR="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/synthetic_data/HF_FORMAT/${DATASET_SLUG}/${MODEL_SLUG}/${SAMPLE_SIZE}samples.jsonl"
JSONL_OUTPUT_PATH="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/synthetic_data/JSONL_FORMAT/${DATASET_SLUG}/${MODEL_SLUG}/${SAMPLE_SIZE}samples.jsonl"

make_directory

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "Model:         $MODEL_NAME"
echo "Dataset:       $DATASET_NAME"
echo "Config:        $DATASET_CONFIG_NAME"
echo "Sample size:   $SAMPLE_SIZE"
echo "Max tokens:  $MAX_TOKEN_LENGTH"
echo "Generations:   $NUM_GENERATIONS_PER_PERSONA"
echo "Seed:          $SEED"
echo "HF Output:     $OUTPUT_DIR"
echo "Jsonl output:  $JSONL_OUTPUT_PATH"
echo "Start time:    $(date)"
echo "HF_HOME:       $HF_HOME"
echo "========================================"

python synthetic_data/api_synthesize.py \
    --model_name "$MODEL_NAME" \
    --sample_size $SAMPLE_SIZE \
    --generations_per_persona $NUM_GENERATIONS_PER_PERSONA \
    --dataset_name "$DATASET_NAME" \
    --dataset_config "$DATASET_CONFIG_NAME" \
    --hf_output_path "$OUTPUT_DIR" \
    --split $SPLIT \
    --jsonl_output_path "$JSONL_OUTPUT_PATH" \
    --temperature $TEMPERATURE \
    --max_token_length $MAX_TOKEN_LENGTH \
    --top_p $TOP_P \
    --seed $SEED \
    --api_key_path "$API_KEY_PATH" 
