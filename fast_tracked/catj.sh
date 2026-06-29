#!/bin/bash
#SBATCH --job-name=api_cat_jsonl
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=0-02:00:00
#SBATCH --mem=16G
#SBATCH --output=/n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails/fast_tracked/logs/%x_%j.out
#SBATCH --error=/n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails/fast_tracked/logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu

set -euo pipefail

env_directory_setup() {
    module purge
    module load python
    mamba activate ossenv
    cd /n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails
}

env_directory_setup

SAMPLE_SIZE=5000
GENERATIONS_PER_INPUT=1
SEED=42
TEMPERATURE=0.6
TOP_P=1.0
MAX_TOKEN_LENGTH=32

MODEL_NAME="gpt-4.1-mini"
MODEL_SLUG="${MODEL_NAME//\//_}"

API_KEY_PATH="/n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails/fast_tracked/api_key.txt"

INPUT_JSONL_PATH="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/synthetic_data/JSONL_FORMAT/proj-persona_PersonaHub/openai_gpt-oss-120b/output_cleaned.jsonl"

JSONL_OUTPUT_PATH="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/categorized_input/JSONL_FORMAT/synthetic_data/${MODEL_SLUG}/${SAMPLE_SIZE}samples.jsonl"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$JSONL_OUTPUT_PATH")"
mkdir -p "fast_tracked/logs"

echo "========================================"
echo "Job ID:        ${SLURM_JOB_ID:-NA}"
echo "Node:          ${SLURMD_NODENAME:-NA}"
echo "Model:         $MODEL_NAME"
echo "Input JSONL:   $INPUT_JSONL_PATH"
echo "Sample size:   $SAMPLE_SIZE"
echo "Max tokens:    $MAX_TOKEN_LENGTH"
echo "Generations:   $GENERATIONS_PER_INPUT"
echo "Seed:          $SEED"
echo "HF Output:     $OUTPUT_DIR"
echo "JSONL output:  $JSONL_OUTPUT_PATH"
echo "Start time:    $(date)"
echo "========================================"

python fast_tracked/jsonl_api_categorize_input.py \
    --model_name "$MODEL_NAME" \
    --input_jsonl_path "$INPUT_JSONL_PATH" \
    --jsonl_output_path "$JSONL_OUTPUT_PATH" \
    --sample_size "$SAMPLE_SIZE" \
    --generations_per_input "$GENERATIONS_PER_INPUT" \
    --api_key_path "$API_KEY_PATH" \
    --temperature "$TEMPERATURE" \
    --max_token_length "$MAX_TOKEN_LENGTH" \
    --top_p "$TOP_P" \
    --seed "$SEED"