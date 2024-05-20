import argparse
from torchvision import models
import torch
from pnmuap import get_data_loader,get_fooling_rate
from functions import get_uap,prepare_for_model,validate_arguments
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import os
from torchvision.models import vgg19,vgg16,resnet152,googlenet,alexnet,vgg16_bn,vgg19_bn,resnet50,resnet101



download_path = 'TorchHub/'
torch.hub.set_dir(download_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_model', default='vgg19',
                        help='The network of target model, eg. vgg19')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size to use for testing')
    parser.add_argument('--val_dataset_name', default='imagenet',
                        help='The dataset to be used as test')
    parser.add_argument('--delta_size', default=32,
                        help='the size of delta')
    parser.add_argument('--epsilon', default=10/255,
                        help='the infinite norm limitation of UAP')
    parser.add_argument('--seed', default=123,
                        help='random seed')
    parser.add_argument('--uap_path', default='perturbations/uap_resnet152_dataset=imagenet_use_end=True_end_rate=0.6_seed=123_prior=gauss_stepiter=200.npy',
                        help='the path of UAP stored')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    validate_arguments(args.test_model)
    # prepare for loading test models
    print('Prepare for the perturbation and models...')
    uap = get_uap(args.uap_path, device)
    test_model = prepare_for_model(args,args.test_model,device,initialize=True)
    test_loader = get_data_loader(args.val_dataset_name,batch_size=args.batch_size,shuffle=True,analyze=True)
    print(f'Attack the target model: {args.test_model}')
    final_fooling_rate = get_fooling_rate(test_model, torch.clamp(uap, -args.epsilon, args.epsilon), test_loader, device)
    print(f"Final fooling rate of UAP on {args.test_model}: {final_fooling_rate}")
    return

if __name__ == '__main__':
    main()