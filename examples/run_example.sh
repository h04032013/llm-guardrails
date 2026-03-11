#!/bin/bash
#SBATCH --job-name=read_data
#SBATCH --account=kempner_dam_lab
#SBATCH --partition=kempner
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --mem=16G
#SBATCH --output=files/output.out
#SBATCH --error=files/error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hdiaz@g.harvard.edu

cd /n/holylabs/LABS/dam_lab/Users/hdiaz/llm-guardrails 

module purge
module load Mambaforge
module load cuda cudnn
mamba activate ossenv

data="/n/netscratch/dam_lab/Lab/hdiaz/guardrail_data/sampled_data.parquet"

PYTHONPATH=. python examples/data_example.py --data $data \