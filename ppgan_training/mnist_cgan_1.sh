#!/bin/bash
python main.py -m normal \
               -ds mnist \
               -mt fc \
               -at leaky \
               -it 50000 \
               -its 100 \
               -opt sgd \
               -lr 0.1 \
               -bs 32 \
               --device cuda \
               -ns 1 \
               -p exp_type_4_0 \
               > logs/type_4_0.txt
               
               