#!/bin/bash
python main.py -m hybrid \
               -ds celeba \
               -mt cnn \
               -at leaky \
               -it 50000 \
               -its 100 \
               -opt sgd \
               -lr 0.002 \
               -bs 32 \
               -ns 2 \
               --device cuda \
               -p exp_4
               
               