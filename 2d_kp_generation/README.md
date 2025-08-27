## Environments

```
conda env create -f environment.yaml
conda activate audio2kp
```



## Data Preparation

#### Weights

Download the pretrained [checkpoint](https://drive.google.com/drive/folders/1sFpj6_is0F8Eq2QLARABjL183dxqGhYc?usp=sharing) and save it in the checkpoints folder like:

```
pretrained_weights
|-- model_weights
|-- wav2vec2-base-960h
```

#### Datasets
We randomly sample 16 cases from the test set.
Download the sample dataset for [here](https://drive.google.com/drive/folders/1nMyT267GoZLp6rRJsKxPpjDbD71WJl1A?usp=sharing)



## Inference

#### Sample Dataset

Refer to  [commands/test_sample.sh]()



## Train

#### Sample Dataset

Refer to  [commands/train_sample.sh]()



## Acknowledgements

Thanks to [mdm](https://github.com/GuyTevet/motion-diffusion-model), [mas](https://github.com/roykapon/MAS), our code is built upon their work.