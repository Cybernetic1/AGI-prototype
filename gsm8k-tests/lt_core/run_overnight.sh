#!/bin/bash
cd /home/yky/misc-programs/AGI-prototype/gsm8k-tests/lt_core/
nohup /home/yky/misc-programs/AGI-prototype/venv/bin/python train_dln_full.py > training.log 2>&1 &
echo "Training started in background! PID: $!"
echo "You can check progress with: tail -f gsm8k-tests/lt_core/training.log"
