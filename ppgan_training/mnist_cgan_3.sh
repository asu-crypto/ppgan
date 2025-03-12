#!/bin/bash
python main.py -m hybrid_cgan \
               -ds mnist \
               -mt fc \
               -at leaky \
               -it 50000 \
               -its 100 \
               -opt sgd \
               -lr 0.1 \
               -bs 32 \
               --device cuda \
               -ns 3 \
               -p exp_cgan_type_4_2 \
               > logs/cgan_type_4_2.txt
               
               
