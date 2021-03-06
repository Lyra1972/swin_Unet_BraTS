import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_BraTS import BraTS_dataset
from datasets.dataset_BraTS3D import BraTS_dataset3D
from utils import test_single_volume
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.vision_transformer import SwinUnet3D as ViT_seg3D
from trainer import trainer_BraTS, trainer_BraTS3D
from config import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='data', help='root dir for validation volume data')  
                    # volume_path = base_dir in BraTS_dataset, default='../data/BraTS/test_vol_h5'
parser.add_argument('--dataset', type=str,
                    default='BraTS', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--in_chans', type=int, default=4, help='input channels')
parser.add_argument('--list_dir', type=str,
                    default='./lists/list_BraTS', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir')   
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--mode', type=str,  default='twoD', help='model mode, twoD/threeD')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
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
if args.dataset == "BraTS":
    args.volume_path = os.path.join(args.volume_path, "BraTS/test__")  # [test_vol_h5, test_new]/[test__, test_vol]
if args.dataset == "BraTS_3D":
    args.volume_path = os.path.join(args.volume_path, "BraTS_3D/test_vol_3D")  # [test_vol_h5, test_new]/[test__, test_vol]
config = get_config(args)


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir) 
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # image.shape == label.shape = (1, 4, 137, 224, 224) or (1, 137, 224, 224)
        h, w = sampled_batch["image"].size()[-2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        # print('0000000',image.shape)
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, mode=args.mode, z_spacing=args.z_spacing,
                                        patch_size=[args.img_size, args.img_size], test_save_path=test_save_path, case=case_name, )
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


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

    dataset_config = {
        'BraTS': {
            'Dataset': BraTS_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/list_BraTS',
            'num_classes': 4,
            'z_spacing': 1,
            'img_size': 224,
            'mode': 'twoD',
        },
        'BraTS_3D': {
            'Dataset': BraTS_dataset3D,
            'volume_path': args.volume_path,
            'list_dir': './lists/list_BraTS3D',
            'num_classes': 4,
            'z_spacing': 1,
            'img_size': 160,
            'mode': 'threeD',
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.img_size = dataset_config[dataset_name]['img_size']
    args.mode = dataset_config[dataset_name]['mode']
    args.is_pretrain = True
    
    if args.mode == 'twoD':
        net = ViT_seg(config, in_chans=args.in_chans, num_classes=args.num_classes).cuda()
    elif args.mode == 'threeD':
        net = ViT_seg3D(config, in_chans=args.in_chans,num_classes=args.num_classes).cuda()

    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet",msg)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


