# Privacy Preserving Generative Adversarial Network

## CelebA dataset
Download the cropped 64x64 CelebA dataset for training at [link]()

## PPGAN training
How to install:
```
conda create -n ppgan python=3.10
conda activate ppgan
cd crypten
pip install .
cp default.yaml ~/anaconda3/envs/ppgan/lib/python3.10/site-packages/configs/default.yaml
```

How to run experiments:

***CelebA:***
```
cd ppgan_training
./exp_{EXPERIMENT NUMBER}
```
Where EXPERIMENT NUMBER indicates which configurations we want to run, experiment 1 indicates plaintext GAN training, experiment 2 indicates private D GAN training, experiment 3,4,5 indicate 3,2,1-layer secure D respectively.

***MNIST:***
```
cd ppgan_training
./mnist_{EXPERIMENT NUMBER}
```
Where EXPERIMENT NUMBER indicates which configurations we want to run, 1 indicates 

## PPGAN Analysis
How to run the analysis:
```
cd ppgan_analysis
python main.py -ds celeba -bsr 1 -bsf 1 -ns NUM_SECURE > log.txt
```
where NUM_SECURE indicates number of secure layer we intend to test, ranging from 1 to 3. with NUM_SECURE=0, we have Federated Learning protocol where no layer in the network are secure.
