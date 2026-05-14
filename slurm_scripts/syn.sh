#!/bin/bash
#SBATCH --job-name=api_syn_10x5
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:30:00
#SBATCH --mem=40G
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
export HF_HOME="/n/hgf_new_hub"

SAMPLE_SIZE=10
NUM_GENERATIONS_PER_PERSONA=5
SEED=42
TEMPERATURE=1.0
TOP_P=1.0
MAX_TOKENS=512

DATASET_SLUG="${DATASET_NAME//\//_}"
MODEL_NAME="gpt-4.1-mini"
MODEL_SLUG="${MODEL_NAME//\//_}"
TENSOR_PARALLEL_SIZE=1
API_KEY_PATH="/n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails/files/logs/api_key.txt"
OUTPUT_DIR="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/categorized_input/HF_FORMAT/${DATASET_SLUG}/${MODEL_SLUG}/${SAMPLE_SIZE}samples/${SPLIT}_${TEXT_COLUMN}.jsonl"
JSONL_OUTPUT_PATH="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/synthetic_data/JSONL_FORMAT/${DATASET_SLUG}/${MODEL_SLUG}/${SAMPLE_SIZE}samples/${SPLIT}_${TEXT_COLUMN}.jsonl"

make_directory

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "Model:         $MODEL_NAME"
echo "Dataset:       $DATASET_NAME"
echo "Config:        $DATASET_CONFIG_NAME"
echo "Sample size:   $SAMPLE_SIZE"
echo "Max tokens:     $MAX_TOKENS"
echo "Generations:   $NUM_GENERATIONS_PER_PERSONA"
echo "Seed:          $SEED"
echo "Output:        $OUTPUT_DIR"
echo "Jsonl output:   $JSONL_OUTPUT_PATH"
echo "Start time:    $(date)"
echo "HF_HOME:       $HF_HOME"
echo "========================================"

python generate/math/synthesize_questions.py \
    --model_name "$MODEL_NAME" \
    --sample_size $SAMPLE_SIZE \
    --generations_per_persona $NUM_GENERATIONS_PER_PERSONA \
    --persona_source_dataset "$DATASET_NAME" \
    --persona_source_dataset_config_name "$DATASET_CONFIG_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --temperature $TEMPERATURE \
    --max_tokens $MAX_TOKENS \
    --top_p $TOP_P \
    --seed $SEED
