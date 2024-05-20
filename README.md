# Progressive Neuron Maximization UAP






## Dependencies

This repo is tested with pytorch<=1.12.0, python<=3.6.13.
Install all python packages using following command:
```
pip install -r requirements.txt
```

## Usage Instructions

### 1. Preparation

ImageNet validation set:
   Load the parameters of pretrained models with PyTorch, download ImageNet dataset from [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
- `TorchHub` : the directory saves PyTorch pretrained model parameters.
- `dataset` : the directory contains the datasets.
- `perturbations` : the directory stores the UAP crafted by universal attacks. 



### 2. Training

For example,run the following command:

```
python train.py --surrogate_model vgg16 --target_model vgg16 --val_dataset_name imagenet --use_end True --end_rate 0.8 --step_iter 100
```
This will start a training to craft a UAP from the surrogate model vgg16 and attack the target model vgg16 on ImageNet with PNM-UAP method.


### 3. Testing
After a UAP is generated and saved on the directory `perturbations`, you can also load the UAP to attack other models:
```
python attack_test.py --test_model vgg19 --val_dataset_name imagenet --uap_path perturbations/uap_vgg16_dataset=imagenet_use_end=True_end_rate=0.8_seed=123_prior=gauss_stepiter=100.npy
```
This will load the UAP made by vgg16 from `perturbations` and attack the target model vgg19 on imagenet.


## Acknowledgements
The code refers to  [TRM-UAP](https://github.com/RandolphCarter0/TRMUAP).

We thank the authors for sharing sincerely.


## Contact

Yuxiang Sun: [syx1191@whu.edu.cn](mailto:syx1191@whu.edu.cn)

