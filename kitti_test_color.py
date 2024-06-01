'''
Doyeon Kim, 2022
'''

import os
import cv2
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import utils.logging as logging
import utils.metrics as metrics
from models.model import CHFNet
from dataset.base_dataset import get_dataset

from PIL import Image
import my_utils
import time
import argparse

class kitti_TestOptions():
    def __init__(self):
        pass

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--gpu_or_cpu', type=str, default='gpu')
        # parser.add_argument('--data_path',    type=str, default='/data/ssd1/') # '../nyu_depth_v2'
        parser.add_argument('--data_path', type=str, default='../../Datasets/kitti')
        parser.add_argument('--dataset', type=str, default='kitti',
                            choices=['nyudepthv2', 'kitti', 'imagepath'])
        # depth configs
        parser.add_argument('--max_depth', type=float, default=80.0)
        parser.add_argument('--max_depth_eval', type=float, default=80.0)#80.0
        parser.add_argument('--min_depth_eval', type=float, default=1e-3)
        # parser.add_argument('--do_kb_crop',     type=int, default=1)
        parser.add_argument('--do_kb_crop', type=int, default=0)#测试不用切割
        parser.add_argument('--garg_crop', default=True)
        # parser.add_argument('--kitti_crop', type=str, default=None,
        # choices=['garg_crop', 'eigen_crop'])
        parser.add_argument('--kitti_crop', type=str, default='eigen_crop')
        parser.add_argument('--eigen_crop', default=False)

        parser.add_argument('--result_dir_est', type=str, default='./kitti_result/est',
                            help='save result images into result_dir_est/exp_name')
        parser.add_argument('--result_dir_att4', type=str, default='./kitti_result/att4',
                            help='save result images into result_dir_att4/exp_name')
        parser.add_argument('--result_dir_att3', type=str, default='./kitti_result/att3',
                            help='save result images into result_dir_att3/exp_name')
        parser.add_argument('--result_dir_att2', type=str, default='./kitti_result/att2',
                            help='save result images into result_dir_att2/exp_name')
        parser.add_argument('--result_dir_att1', type=str, default='./kitti_result/att1',
                            help='save result images into result_dir_att1/exp_name')
        parser.add_argument('--visual_dir_est', type=str, default='./kitti_visual/est',
                            help='save result images into visual_dir/exp_name')
        parser.add_argument('--visual_dir_att4', type=str, default='./kitti_visual/att4',
                            help='save result images into visual_dir/exp_name')
        parser.add_argument('--visual_dir_att3', type=str, default='./kitti_visual/att3',
                            help='save result images into visual_dir/exp_name')
        parser.add_argument('--visual_dir_att2', type=str, default='./kitti_visual/att2',
                            help='save result images into visual_dir/exp_name')
        parser.add_argument('--visual_dir_att1', type=str, default='./kitti_visual/att1',
                            help='save result images into visual_dir/exp_name')
        parser.add_argument('--ckpt_dir',   type=str,
                            default='/media/6THDD/chenhan/my_code/Release/CHFNet/logs/final_result/kitti/epoch_20_global_step_77000_best.pt',
                            help='load ckpt path')

        parser.add_argument('--do_evaluate', default=True) # True
        parser.add_argument('--save_eval_pngs',default=False)
        parser.add_argument('--save_visualize', default=False)
        parser.add_argument('--batch_size', default=2)

        return parser

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


opt = kitti_TestOptions()
args = opt.initialize().parse_args()
print(args)

if args.gpu_or_cpu == 'gpu':
    device = torch.device('cuda')
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

if args.save_eval_pngs or args.save_visualize:

    result_path = []
    visual_path = []
    result_path.append(args.result_dir_est)
    result_path.append(args.result_dir_att4)
    result_path.append(args.result_dir_att3)
    result_path.append(args.result_dir_att2)
    result_path.append(args.result_dir_att1)
    visual_path.append(args.visual_dir_est)
    visual_path.append(args.visual_dir_att4)
    visual_path.append(args.visual_dir_att3)
    visual_path.append(args.visual_dir_att2)
    visual_path.append(args.visual_dir_att1)
    for i in range(len(result_path)):
        logging.check_and_make_dirs(result_path[i])
        logging.check_and_make_dirs(visual_path[i])
        print("Saving result images in to %s" % result_path[i])
        print("Saving result images in to %s" % visual_path[i])

if args.do_evaluate:
    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

print("\n1. Define Model")
model = CHFNet(max_depth=args.max_depth, is_train=False).to(device)

model_weight = torch.load(args.ckpt_dir)['model']
if 'module' in next(iter(model_weight.items()))[0]:
    model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
model.load_state_dict(model_weight,False)
model.eval()

num_params = sum([np.prod(p.size()) for p in model.parameters()])
print("Total number of parameters: {}".format(num_params))

print("\n2. Define Dataloader")
if args.dataset == 'imagepath': 
    dataset_kwargs = {'dataset_name': 'ImagePath', 'data_path': args.data_path}
else:
    dataset_kwargs = {'data_path': args.data_path, 'dataset_name': args.dataset,
                      'is_train': False}

test_dataset = get_dataset(**dataset_kwargs)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                         pin_memory=True)

print("\n3. Inference & Evaluate")
metrics = my_utils.RunningAverageDict()
res = []
for batch_idx, batch in enumerate(test_loader):
    input_RGB = batch['image'].to(device)
    filename = batch['filename']

    with torch.no_grad():
        start = time.time()
        pred, att4, att3, att2, att1 = model(input_RGB)
        torch.cuda.synchronize()
        end = time.time()
        res.append(end-start)
        output = []
        output.append(pred)
        output.append(att4)
        output.append(att3)
        output.append(att2)
        output.append(att1)
    for i in range(args.batch_size):
        if args.do_evaluate:
            depth_gt    = batch['depth'][i].to(device)
            pred_d      = output[0][i]
            pred_d      = pred_d.squeeze().cpu().numpy()
            gt_depth    = depth_gt.squeeze().cpu().numpy()
            """ pred中有一些奇异值, 将其统一 """
            pred_d[pred_d < args.min_depth_eval] = args.min_depth_eval
            pred_d[pred_d > args.max_depth_eval] = args.min_depth_eval
            pred_d[np.isinf(pred_d)] = args.min_depth_eval
            pred_d[np.isnan(pred_d)] = args.min_depth_eval
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < 80)
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
            metrics.update(my_utils.compute_errors(gt_depth[valid_mask], pred_d[valid_mask]))

        if args.save_eval_pngs:
            for i in range(len(output)):
                save_path = os.path.join(result_path[i], filename[0])
                if save_path.split('.')[-1] == 'jpg':
                    save_path = save_path.replace('jpg', 'png')
                pred_d = output[i].cpu().numpy().squeeze()
                if args.dataset == 'nyudepthv2':
                    pred_d = pred_d * 1000.0
                    cv2.imwrite(save_path, pred_d.astype(np.uint16),
                                [cv2.IMWRITE_PNG_COMPRESSION, 0])
                else:
                    pred_d = pred_d * 256.0
                    cv2.imwrite(save_path, pred_d.astype(np.uint16),[cv2.IMWRITE_PNG_COMPRESSION, 0])

        if args.save_visualize:
            for i in range(len(output)):
                save_path = os.path.join(visual_path[i], filename[0])

                pred_d_numpy = output[i].cpu().numpy().squeeze()
                if args.dataset == 'nyudepthv2':
                    pred_d_numpy = pred_d_numpy[45:472, 43:608]
                
                pred_d_numpy-=pred_d_numpy.min()
                pred_d_numpy = (pred_d_numpy / (pred_d_numpy.max()-pred_d_numpy.min())) * 255
                pred_d_numpy = 255 - pred_d_numpy
                im_color = cv2.applyColorMap(cv2.convertScaleAbs(pred_d_numpy, alpha=1), cv2.COLORMAP_JET)
                im = Image.fromarray(im_color)
                im.save(save_path)

time_sum = 0
for i in res:
    time_sum += i
print("runtime: %f"%(time_sum/len(res)))
print("FPS: %f"%(1.0/(time_sum/len(res))))

if args.do_evaluate:
    result_metrics = metrics.get_value()

    display_result = logging.display_result(result_metrics)
    if args.kitti_crop:
        print("\nCrop Method: ", args.kitti_crop)
    print(display_result)

print("Done")



