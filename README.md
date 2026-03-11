# llm-guardrails

## Setup
1. Resquesting an interactive session on the Kempner Partition:
```bash
salloc -p kempner_interactive --account=kempner_dam_lab --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=16G --gres=gpu:1 -t 00-03:00:00
```

2. Creating a virtual environment using Conda:
```bash
module load python/3.10.12-fasrc01
conda create --name ossenv python=3.12 pip numpy
conda activate ossenv
```

Next, install the required packages:
```bash
pip install --upgrade uv
uv pip install https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl
uv pip install --pre vllm \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match

```
Warning: this might take sometime;

## Directory Structure
```
llm-guardrails/
├── examples/ #demo scripts for prototyping 
    ├── data_example.py #load in extracted user input from anonymous reddit data
    ├── run_example.sh
└── files/               # gitignored, outputs + runtime artifacts
```
The `files` should contain all the directories for the outputs, results, and logs.

## Example Usage
To run the vLLM example on interactive node, execute:
```bash
TBD 
```
