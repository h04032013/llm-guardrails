#!/bin/bash
#SBATCH --job-name=shen_single_500
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

make_directory() {
    mkdir -p "$(dirname "$JSONL_OUTPUT_PATH")"
    mkdir -p "fast_tracked/logs"
}

env_directory_setup

SAMPLE_SIZE=500
GENERATIONS_PER_INPUT=1
SEED=42
TEMPERATURE=0.6
TOP_P=1.0
MAX_TOKEN_LENGTH=512

MODEL_NAME="gpt-4o-mini"
MODEL_SLUG="gpt-4o-mini"

TEXT_COLUMN="input_text"
LABEL_COLUMN="response"

API_KEY_PATH="/n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails/fast_tracked/api_key.txt"

INPUT_JSONL_PATH="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/categorized_input/JSONL_FORMAT/ShenLab_MentalChat16K/gpt-4.1-mini/500samples.jsonl"

JSONL_OUTPUT_PATH="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/generated_responses/ShenLab_MentalChat16K/${MODEL_SLUG}/${SAMPLE_SIZE}samples_responses.jsonl"

make_directory

echo "========================================"
echo "Job ID:        ${SLURM_JOB_ID:-NA}"
echo "Node:          ${SLURMD_NODENAME:-NA}"
echo "Model:         $MODEL_NAME"
echo "Input JSONL:   $INPUT_JSONL_PATH"
echo "Output JSONL:  $JSONL_OUTPUT_PATH"
echo "Sample size:   $SAMPLE_SIZE"
echo "Max tokens:    $MAX_TOKEN_LENGTH"
echo "Generations:   $GENERATIONS_PER_INPUT"
echo "Seed:          $SEED"
echo "Start time:    $(date)"
echo "========================================"

python fast_tracked/api_single_turn.py \
    --model_name "$MODEL_NAME" \
    --input_jsonl_path "$INPUT_JSONL_PATH" \
    --jsonl_output_path "$JSONL_OUTPUT_PATH" \
    --text_column "$TEXT_COLUMN" \
    --label_column "$LABEL_COLUMN" \
    --sample_size "$SAMPLE_SIZE" \
    --generations_per_input "$GENERATIONS_PER_INPUT" \
    --api_key_path "$API_KEY_PATH" \
    --temperature "$TEMPERATURE" \
    --max_token_length "$MAX_TOKEN_LENGTH" \
    --top_p "$TOP_P" \
    --seed "$SEED"