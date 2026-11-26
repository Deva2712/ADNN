#!/bin/bash
#SBATCH -J adnn-gpu
#SBATCH -p gpu_student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 10:00:00
#SBATCH -o /dgxa_home/se23ucse019/adnn/logs/worker-%j.out
# GPU worker: runs queue/*.sh in name order, one at a time. touch queue/STOP to end.
cd ~/adnn
echo "job=$SLURM_JOB_ID node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} start=$(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
while [ ! -f queue/STOP ]; do
  f=$(ls queue/*.sh 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    n=$(basename "$f" .sh); echo "=== $n start $(date)"
    bash "$f" > "logs/$n.log" 2>&1; echo "=== $n exit=$? $(date)"
    mv "$f" done/
  else sleep 5; fi
done
echo "stopped $(date)"
