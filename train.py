import argparse

import matplotlib.pyplot as plt
from torchvision import models
import torch
from pnmuap import *
from functions import validate_arguments

import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import os



download_path = 'TorchHub/'
torch.hub.set_dir(download_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--step_iter', type=int, default=100,
                        help='The number of iterations needed for ratio increase')
    parser.add_argument('--surrogate_model', default='vgg19',
                        help='The surrogate_model eg. vgg19')
    parser.add_argument('--target_model', default='vgg19',
                        help='The target model eg. vgg19')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size to use for training and testing')
    parser.add_argument('--patience_interval', type=int, default=5,
                        help='The number of iterations to wait to verify convergence')
    parser.add_argument('--val_dataset_name', default='imagenet',choices=['imagenet'],
                        help='The dataset to be used as test')

    parser.add_argument('--start_rate', default=0.1, type=float,
                        help='start proportion of layer used')
    parser.add_argument('--end_rate', default=0.8, type=float,
                        help='end proportion of conv layer used')
    parser.add_argument('--use_end', default=True, type=lambda x: x.lower() == 'true',
                        help='use PNM method or not')
    parser.add_argument('--seed', default=123,
                        help='random seed')
    parser.add_argument('--epsilon', default=10/255,
                        help='the infinite norm limitation of UAP')
    parser.add_argument('--delta_size', default=32,
                        help='the size of delta')
    parser.add_argument('--uap_lr', default=0.1,
                        help='the leraning rate of UAP')
    parser.add_argument('--prior', default='gauss',choices=['gauss','jigsaw','None'],
                        help='the range prior of perturbations')
    parser.add_argument('--prior_batch', default=1,
                        help='the batch size of prior')
    parser.add_argument('--std', default=10,
                        help='initialize the standard deviation of gaussian noise')
    parser.add_argument('--fre', default=1,
                        help='initialize the frequency of jigsaw image')
    parser.add_argument('--uap_path', default=None,
                        help='the path of UAP')
    parser.add_argument('--gauss_t0', default=400,
                        help='the threshold to adjust the increasing rate of standard deviation(gauss)')
    parser.add_argument('--gauss_gamma', default=10,
                        help='the step size(gauss)')
    parser.add_argument('--jigsaw_t0', default=600,
                        help='the threshold to adjust the increasing rate of standard deviation(jigsaw)')
    parser.add_argument('--jigsaw_gamma', default=1,
                        help='the step size(jigsaw)')
    parser.add_argument('--jigsaw_end_iter', default=4200,
                        help='the iterations which stop the increment of frequency(jigsaw)')

    #best set-up    end_rate    prior strategy      step_iter:
    #alexnet        0.8         gauss               50
    #vgg16          0.8         gauss               100
    #vgg19          0.8         gauss               100
    #resnet152      0.6         gauss               200
    #googlnet       0.8         jigsaw              150


    args = parser.parse_args()
    validate_arguments(args.surrogate_model)
    validate_arguments(args.target_model)

    if torch.cuda.is_available():
        # set the random seed
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    cudnn.benchmark = True

    #for test

    # perpare for the surrogate model and target model
    model = prepare_for_model(args,args.surrogate_model,device,initialize=True)
    model.eval()

    # print(model)
    # raise ValueError()
    target_model = prepare_for_model(args,args.target_model,device,initialize=False)
    # craft UAP or loading it from the file
    if args.uap_path == None:
        uap = PNM_UAP(model, args, device,prior=args.prior)
        filename = f"perturbations/uap_{args.surrogate_model}_dataset={args.val_dataset_name}_use_end={args.use_end}" \
                   f"_end_rate={args.end_rate}_seed={args.seed}_prior={args.prior}_stepiter={args.step_iter}"
        np.save(filename, uap.cpu().detach().numpy())
        print(f'the UAP of surrogate model {args.surrogate_model} is crfted.')
    else:
        uap = get_uap(args.uap_path,device)

    test_loader = get_data_loader(args.val_dataset_name, batch_size=args.batch_size, shuffle=True, analyze=True)
    final_fooling_rate = get_fooling_rate(target_model,torch.clamp(uap,-args.epsilon,args.epsilon),test_loader,device)
    print(f'the FR of UAP ({args.surrogate_model}) on ({args.target_model}) is {final_fooling_rate}')
    print('finish')

if __name__ == '__main__':
    main()