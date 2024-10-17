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
    return cfg