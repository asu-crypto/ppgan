#!/bin/bash
max=10
for i in `seq 1 $max`
do
    python gen_image2.py -c $i && python faster-pytorch-fid/fid_score_gpu.py tmp_generated_2 mnist_represent.npz >> fid_log_mnist_1l
done