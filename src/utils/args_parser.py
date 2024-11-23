import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="/home/user/fedmu/configs/isic/run_conf.yaml", help='config file')
    parser.add_argument('--eval_only', type=int, default=False, help='whether only test')
    parser.add_argument('--resume_path', type=str, default="", help='saved checkpoin')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--gpu', type=str, default='1', help='whether use gpu')
    parser.add_argument('--val_interval', type=int, default=1, help='valdation interval')
    parser.add_argument('--amp', type=int, default=False, help='mixed precision')
    parser.add_argument('--path_checkpoint', type=str, default=None, help='path of the checkpoint of the model')
    parser.add_argument('--deterministic', action='store_true', help='whether use deterministic training, default is False')

    parser.add_argument('--run_name', type=str, default=None, help='run name of wandb')    
    parser.add_argument('--train_path', type=str, default=None, help='train path for localized training')
    parser.add_argument('--test_path', type=str, default=None, help='test path for localized training')
    parser.add_argument('--trainer', type=str, default='supervised', help='trainer type: supervised, semi')
    parser.add_argument('--labeled_clients', type=str, nargs='+', default=None, help='labeled clients')
    parser.add_argument('--unseen_client', type=str, default=None, help='unseen client')
    parser.add_argument('--use_ga', action='store_true', help='Whether to use GA, default is False')
    parser.add_argument('--fp_rate', type=float, default=None, help='feature dropping rate')
    parser.add_argument('--feature_loss_weight', type=float, default=None, help='feature loss weight')
    parser.add_argument('--entropy_start_ratio', type=float, default=None, help='uncertain ratio')
    parser.add_argument('--entropy_end_ratio', type=float, default=None, help='uncertain ratio')

    args, unknown = parser.parse_known_args()
    return args

args = args_parser()

def args2cfg(cfg, args):
    run_name = args.run_name
    cfg['wandb']['run_name'] = run_name
    if args.train_path is not None:
        cfg['dataset']['train'] = args.train_path
    
    if args.test_path is not None:
        cfg['dataset']['test'] = args.test_path

    if args.use_ga is not None:
        cfg['fl']['use_ga'] = args.use_ga
        print(args.use_ga)
    
    if args.fp_rate is not None:
        cfg['model']['fp_rate'] = args.fp_rate
    
    if args.feature_loss_weight is not None:
        cfg['train']['feature_loss_weight'] = args.feature_loss_weight

    if args.entropy_start_ratio is not None:
        cfg['train']['entropy_start_ratio'] = args.entropy_start_ratio

    if args.entropy_end_ratio is not None:
        cfg['train']['entropy_end_ratio'] = args.entropy_end_ratio
    
    return cfg