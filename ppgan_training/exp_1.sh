#!/bin/bash
python main.py -m normal \
               -ds celeba \
               -mt cnn \
               -at leaky \
               -it 50000 \
               -its 100 \
               -opt sgd \
               -lr 0.002 \
               -bs 32 \
               --device cuda \
               -p exp_1 \
               > logs/exp_1.txt
               
               