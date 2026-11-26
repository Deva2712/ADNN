#!/bin/bash
#SBATCH -J adnn-sw
#SBATCH -p cpu_student
#SBATCH -c 20
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /dgxa_home/se23ucse019/adnn/logs/sw-%j.out
cd ~/adnn; ./venv/bin/python sw_label.py --procs 20; echo SW_DONE
