
def curriculum_strategy_gauss(iter_num,args):
    if iter_num == 1600 or iter_num == 2400 or iter_num == 3200:
        args.prior_batch = args.prior_batch * 2

    if iter_num % args.step_iter == 0  and args.start_rate < args.end_rate and args.use_end==True:
        args.start_rate=round(args.start_rate+0.05,2)
        # print(f'now is {args.start_rate}')

    if iter_num % args.gauss_t0 == 0 and args.std < 127 and args.surrogate_model not in ['vgg16','vgg19','vgg16_bn','vgg19_bn','resnet101']:
        args.std = args.std + args.gauss_gamma
    if args.std > 127:
        args.std = 127
    return args

def curriculum_strategy_jigsaw(iter_num,args):
    if iter_num == 1600 or iter_num==3200:
        args.prior_batch = args.prior_batch*2
    if iter_num % args.jigsaw_t0 == 0 and iter_num < args.jigsaw_end_iter:
        args.fre = args.fre + args.jigsaw_gamma
    if iter_num % 100 == 0  and args.start_rate < args.end_rate and args.use_end==True:
        args.start_rate=round(args.start_rate+0.05,2)
        # print(f'now is {args.start_rate}')
    return args

def curriculum_strategy_jigsaw_resnet152(iter_num,args):
    if iter_num == 1600 or iter_num==3200:
        args.prior_batch = args.prior_batch*2

    if iter_num < 1000:
        if iter_num % 200 == 0:
            args.fre += 1

    if iter_num >= 1000:  # and iter_num <4200:
        if iter_num % 400 == 0:
            args.fre += 1

    return args

def curriculum_strategy_jigsaw_googlenet(iter_num,args):
    if iter_num == 1600 or iter_num == 2400 or iter_num == 3200:
        args.prior_batch = args.prior_batch * 2
    if iter_num % 400 == 0:
        args.fre += 1
    if args.start_rate==0.4:
        args.start_rate=args.end_rate
    if iter_num % args.step_iter == 0  and args.start_rate < args.end_rate and args.use_end==True:
        args.start_rate=round(args.start_rate+0.05,2)
        # print(f'now is {args.start_rate}')

    if iter_num > 1200 and iter_num <= 3000:

        if iter_num % 600 == 0:
            args.fre += 1

    return args
