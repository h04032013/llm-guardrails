
set -euo pipefail

env_directory_setup() {
    module purge
    module load python
    mamba activate ossenv
    cd /n/llm-guardrails
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
SHUFFLE_SEED=42
TEMPERATURE=0.0
TOP_P=1.0
MAX_TOKENS=4096

DATASET_NAME="proj-persona/PersonaHub"
DATASET_CONFIG_NAME="persona"
DATASET_SLUG="${DATASET_NAME//\//_}"
MODEL_PATH="openai/gpt-oss-20b"
MODEL_SLUG="${MODEL_PATH//\//_}"
TENSOR_PARALLEL_SIZE=1
OUTPUT_DIR="/guardrail_data/synthetic_user_messages/${DATASET_SLUG}/${MODEL_SLUG}/${SAMPLE_SIZE}samples"
JSONL_OUTPUT_PATH="/guardrail_data/categorized_input/jsonl_format/${DATASET_SLUG}/${MODEL_SLUG}/${SAMPLE_SIZE}samples/${SPLIT}_${TEXT_COLUMN}.jsonl"

make_directory

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "Model:         $MODEL_PATH"
echo "Dataset:       $DATASET_NAME"
echo "Config:        $DATASET_CONFIG_NAME"
echo "Sample size:   $SAMPLE_SIZE"
echo "Generations:   $NUM_GENERATIONS_PER_PERSONA"
echo "Seed:          $SEED"
echo "Shuffle seed:  $SHUFFLE_SEED"
echo "Output:        $OUTPUT_DIR"
echo "Start time:    $(date)"
echo "HF_HOME:       $HF_HOME"
echo "========================================"

python generate/math/synthesize_questions.py \
    --model_path "$MODEL_PATH" \
    --sample_size $SAMPLE_SIZE \
    --num_questions_per_persona $NUM_GENERATIONS_PER_PERSONA \
    --persona_source_dataset "$DATASET_NAME" \
    --persona_source_dataset_config_name "$DATASET_CONFIG_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --temperature $TEMPERATURE \
    --max_tokens_questions $MAX_TOKENS_QUESTIONS \
    --top_p $TOP_P \
    --shuffle_seed $SHUFFLE_SEED \
    --seed $SEED
