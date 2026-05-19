#!/bin/bash
#SBATCH --job-name=cat_synthetic_full
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=0-00:15:00
#SBATCH --mem=64G
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

SAMPLE_SIZE=0
DATASET_NAME="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/synthetic_data/HF_FORMAT/proj-persona_PersonaHub/gpt-4.1-mini/2500samples.jsonl"
DATASET_SLUG="${DATASET_NAME//\//_}"
MODEL_PATH="openai/gpt-oss-20b"
SPLIT="train"
MODEL_SLUG="${MODEL_PATH//\//_}"
TEXT_COLUMN="synthetic_input"
OUTPUT_DIR="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/categorized_input/HF_FORMAT/$synthetic_data/${MODEL_SLUG}/${SAMPLE_SIZE}samples"
TEMPERATURE=0.7
TOP_P=1.0
MAX_TOKEN_LENGTH=4096
SEED=42
DATASET_SOURCE="disk" # options: "hub" or "disk"
TENSOR_PARALLEL_SIZE=1
JSONL_OUTPUT_PATH="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/categorized_input/JSONL_FORMAT/synthetic_data/${MODEL_SLUG}/${SAMPLE_SIZE}samples/${SPLIT}_${TEXT_COLUMN}.jsonl"

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

python categorize_text/vllm_categorize_input.py\
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
    --dataset_source "$DATASET_SOURCE" \
    --seed $SEED
