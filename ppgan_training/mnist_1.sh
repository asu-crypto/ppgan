#!/bin/bash
python main.py -m full \
               -ds mnist \
               -mt fc \
               -at relu \
               -it 50000 \
               -its 100 \
               -opt sgd \
               -lr 0.1 \
               -bs 32 \
               --device cuda \
               -ns 3 \
               -p exp_type_2_1 \
               > logs/type_2_1.txt
               
               