#!/bin/bash
#SBATCH --job-name=tulu_synth_5k
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=0-00:45:00
#SBATCH --mem=128G
#SBATCH --output=/n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails/fast_tracked/logs/%x_%j.out
#SBATCH --error=/n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails/fast_tracked/logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu

set -euo pipefail

module purge
module load python
mamba activate ossenv
cd /n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails/fast_tracked
export HF_HOME="/n/netscratch/dam_lab/Lab/hdiaz/hgf_new_hub"

SAMPLE_SIZE=5000
NUM_GENERATIONS_PER_PERSONA=3
SEED=42
TEMPERATURE=1.0
TOP_P=1.0
MAX_TOKEN_LENGTH=4096
DATASET_NAME="proj-persona/PersonaHub"
DATASET_CONFIG_NAME="persona"
TENSOR_PARALLEL_SIZE=2
SPLIT="train"
MAX_MODEL_LENGTH=4096

MODEL_NAME="openai/gpt-oss-120b"
DATASET_SLUG="${DATASET_NAME//\//_}"
MODEL_SLUG="${MODEL_NAME//\//_}"

#OUTPUT_DIR="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/synthetic_data/HF_FORMAT/${DATASET_SLUG}/${MODEL_SLUG}/${SAMPLE_SIZE}samples"
JSONL_OUTPUT_PATH="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/synthetic_data/JSONL_FORMAT/${DATASET_SLUG}/${MODEL_SLUG}/${SAMPLE_SIZE}samples.jsonl"

mkdir -p "$(dirname "$JSONL_OUTPUT_PATH")"

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "Model:         $MODEL_NAME"
echo "Dataset:       $DATASET_NAME"
echo "Sample size:   $SAMPLE_SIZE"
echo "Generations:   $NUM_GENERATIONS_PER_PERSONA"
echo "JSONL output:  $JSONL_OUTPUT_PATH"
echo "HF_HOME:       $HF_HOME"
echo "Start time:    $(date)"
echo "========================================"

python vllm_synthesize.py \
    --model_path "$MODEL_NAME" \
    --sample_size "$SAMPLE_SIZE" \
    --num_questions_per_persona "$NUM_GENERATIONS_PER_PERSONA" \
    --output_path "$JSONL_OUTPUT_PATH" \
    --max_model_len "$MAX_MODEL_LENGTH" \
    --temperature "$TEMPERATURE" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --max_tokens_questions "$MAX_TOKEN_LENGTH" \
    --top_p "$TOP_P" \
    --seed "$SEED" 