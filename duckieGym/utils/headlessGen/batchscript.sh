#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=10G
#SBATCH --time=22:00:00
#SBATCH -o /network/tmp1/courchea/slurm-%j.out

module load python/3.8

virtualenv tmp
source tmp/bin/activate
pip install -r requirements.txt

source startvx.sh

sleep 2

xdpyinfo -display :99 >/dev/null 2>&1 && echo "In use" || echo "Free"

python3 groundtruth.py --steps 1000 --nb-episodes 1200
