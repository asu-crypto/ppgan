#!/bin/bash
python main.py -m hybrid \
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
               -p exp_type_2_6 \
               > logs/type_2_6.txt
               
               