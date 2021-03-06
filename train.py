import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.vision_transformer import SwinUnet3D as ViT_seg3D
from trainer import trainer_BraTS, trainer_BraTS3D
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='data', help='root dir for data')             # '../data/Synapse/train_npz'
parser.add_argument('--dataset', type=str,
                    default='BraTS', help='experiment_name')                    # default='Synapse'
parser.add_argument('--list_dir', type=str,
                    default='./lists/list_BraTS', help='list dir')              # default='./lists/lists_Synapse'
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')                # default=9
parser.add_argument('--output_dir', type=str, default='traning/', help='output dir') 
parser.add_argument('--save_name', type=str,
                    default=' ', help='save name') 
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--in_chans', type=int,  default=4, help='input channels')
parser.add_argument('--mode', type=str,  default='twoD', help='model mode, twoD/threeD')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file') # required=True
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "BraTS" :      # Synapse
    args.root_path = os.path.join(args.root_path, "BraTS/train_npz")   # train_npz
elif args.dataset == "BraTS_3D":  
    args.root_path = os.path.join(args.root_path, "BraTS_3D/train_npz")
config = get_config(args)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'BraTS': {
            'root_path': args.root_path,
            'list_dir': './lists/list_BraTS',
            #'root_path': './data/BraTS',
            'num_classes': 4,
        },
        'BraTS_3D': {
            'root_path': args.root_path,
            'list_dir': './lists/list_BraTS3D',
            #'root_path': './data/BraTS_3D',
            'num_classes': 4, 
        },
    }
    
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.mode == 'twoD':
        net = ViT_seg(config, in_chans=args.in_chans, num_classes=args.num_classes).cuda()
    elif args.mode == 'threeD':
        net = ViT_seg3D(config, in_chans=args.in_chans,num_classes=args.num_classes).cuda()
    net.load_from(config)

    trainer = {'BraTS': trainer_BraTS,'BraTS_3D':trainer_BraTS3D}
    trainer[dataset_name](args, net, args.output_dir, args.save_name)