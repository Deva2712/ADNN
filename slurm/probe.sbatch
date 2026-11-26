#!/bin/bash
#SBATCH -J adnn-probe
#SBATCH -p cpu_student
#SBATCH -c 24
#SBATCH --mem=96G
#SBATCH -t 01:30:00
#SBATCH -o /dgxa_home/se23ucse019/adnn/logs/probe-%x-%j.out
cd ~/adnn; time ./venv/bin/python probe.py --model $1 --jobs 24; echo PROBE_DONE
