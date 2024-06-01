""" 
这个版本 
"""

import os

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torchvision import transforms
from models.model import CHFNet

import my_utils
from my_utils import RunningAverage
from utils.criterion import SiLogLoss,SmoothLoss,gradientLoss
from dataset.base_dataset import get_dataset
from configs.kitti_options import KITTI_Options

from tqdm import tqdm
from datetime import datetime
import socket
import time

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log','log10', 'silog'] 
loss_weights = [1, 1, 1, 1]

def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def validate(args, model, test_loader, criterion_ueff):
    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = my_utils.RunningAverageDict()
        
        for  batch in tqdm(test_loader):

            img = batch['image'].cuda() #####
            depth = batch['depth'].cuda() ##############

            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            depth= depth.unsqueeze(0)

            pred, _, _, _,_  = model(img)

            # mask = depth > args.min_depth_eval
            l_dense = criterion_ueff(pred, depth)
            val_si.append(l_dense.item())

            pred = pred.squeeze().cpu().numpy()
            
            pred[pred < args.min_depth_eval] = args.min_depth_eval 
            pred[pred > args.max_depth_eval] = args.min_depth_eval
            pred[np.isinf(pred)] = args.min_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
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
            metrics.update(my_utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_si

# #############
opt = KITTI_Options()
args = opt.initialize().parse_args()

np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

log_dir = os.path.join(os.path.abspath(os.getcwd()), 'logs', 
                       datetime.now().strftime('%b%d_%H-%M-%S_') + socket.gethostname())
os.makedirs(log_dir)

writer = SummaryWriter(log_dir + '/' + args.log_directory + '/summaries', flush_secs=30)

eval_summary_path = os.path.join(log_dir, "eval_online")
eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

model = CHFNet(max_depth=args.max_depth, is_train=True)

num_params = sum([np.prod(p.size()) for p in model.parameters()])
print("Total number of parameters: {}".format(num_params))

# CPU-GPU agnostic settings
if args.gpu_or_cpu == 'gpu':
    device = torch.device('cuda')
    cudnn.benchmark = True
    model = torch.nn.DataParallel(model)
else:
    device = torch.device('cpu')
model.to(device)

# print(model)

# Dataset setting
dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
if args.dataset == 'nyudepthv2':
    dataset_kwargs['crop_size'] = (448, 576)
elif args.dataset == 'kitti':
    dataset_kwargs['crop_size'] = (352, 704)
else:
    dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

train_dataset = get_dataset(**dataset_kwargs)
val_dataset = get_dataset(**dataset_kwargs, is_train=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                           shuffle=True, num_workers=args.workers,
                                           pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                         pin_memory=True)

# Training settings
criterion_d = SiLogLoss()
criterion_s = SmoothLoss()
criterion_g = gradientLoss().cuda()
optimizer   = optim.Adam(model.parameters(), args.lr)
loss_fun_id = 0 

eval_metrics = ['a1', 'a2', 'a3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'silog', 'log_10']
eval_metrics_low_better = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'silog', 'log_10']

iters = len(train_loader)

best_to_save = dict(a1=0, a2=0, a3=0, abs_rel=np.inf, sq_rel=np.inf, rmse=np.inf, rmse_log=np.inf, silog=np.inf,
                    log_10=np.inf)

total_steps = args.epochs * iters
half_steps = total_steps // 2

start_time = time.time()
duration = 0

num_log_images = args.batch_size

global_step = 0

model.train()
# Perform experiment
for epoch in range(1, args.epochs + 1):

    for batch_idx, batch in enumerate(train_loader):

        before_op_time = time.time()
        for param_group in optimizer.param_groups:
            if global_step < half_steps:
                current_lr = (1e-4 - 3e-5) * (global_step /
                                              half_steps) ** 0.9 + 3e-5
            else:
                current_lr = (3e-5 - 1e-4) * (global_step /
                                              half_steps - 1) ** 0.9 + 1e-4
            param_group['lr'] = current_lr

        img = batch['image'].to(device)
        depth = batch['depth'].to(device).unsqueeze(1)

        pred, att4, att3, att2, att1 = model(img)
        pred[pred > args.max_depth_eval]      = args.max_depth_eval
        optimizer.zero_grad()

        if global_step < half_steps:
            if loss_fun_id == 0:
                loss = criterion_d(pred, depth)+loss_weights[0]*criterion_d(att1, depth)+\
                loss_weights[1]*criterion_d(att2, depth)+loss_weights[2]*criterion_d(att3, depth)+loss_weights[3]*criterion_d(att4, depth)+criterion_s(pred,img)*0.1
            else:
                loss = criterion_d(pred, depth)

        else:
            if loss_fun_id == 0:
                loss = criterion_d(pred, depth)+loss_weights[0]*criterion_d(att1, depth)+\
                loss_weights[1]*criterion_d(att2, depth)+loss_weights[2]*criterion_d(att3, depth)+loss_weights[3]*criterion_d(att4, depth)+criterion_s(pred,img)*0.1
            else:
                loss = criterion_d(pred, depth)

        loss.backward()

        optimizer.step()

        if global_step % 10 == 0:
            print(' [epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, batch_idx,
                                                                                             iters,
                                                                                             global_step,
                                                                                             current_lr, loss))
        if np.isnan(loss.cpu().item()):
            print('NaN in loss occurred. Aborting training.')

        duration += time.time() - before_op_time

        if global_step and global_step % args.log_freq == 0:
            model_params = list(model.parameters())
            var_sum = [var.detach().cpu().sum() for var in model_params if var.requires_grad]
            var_cnt = len(var_sum)
            var_sum = np.sum(var_sum)
            examples_per_sec = args.batch_size / duration * args.log_freq
            duration = 0
            time_sofar = (time.time() - start_time) / 3600
            training_time_left = (total_steps / global_step - 1.0) * time_sofar
            print("{}".format(args.name))
            print_string = 'examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
            print(print_string.format(examples_per_sec, loss, var_sum.item(), var_sum.item() / var_cnt,
                                      time_sofar, training_time_left))

            writer.add_scalar('loss', loss, global_step)
            writer.add_scalar('learning_rate', current_lr, global_step)
            writer.add_scalar('var average', var_sum.item() / var_cnt, global_step)
            depth_gt = torch.where(depth < 1e-3, depth * 0 + 1e3, depth)
            for i in range(num_log_images):
                writer.add_image('depth_gt/image/{}'.format(i), normalize_result(1/depth_gt[i, :, :, :]),global_step)
                writer.add_image('depth_est/image/{}'.format(i), normalize_result(1/pred[i, :, :, :].data),global_step)
                writer.add_image('att4/image/{}'.format(i), normalize_result(1/att4[i, :, :, :].data),global_step)
                writer.add_image('att3/image/{}'.format(i), normalize_result(1/att3[i, :, :, :].data),global_step)
                writer.add_image('att2/image/{}'.format(i), normalize_result(1/att2[i, :, :, :].data),global_step)
                writer.add_image('att1/image/{}'.format(i), normalize_result(1/att1[i, :, :, :].data),global_step)
                writer.add_image('image/image/{}'.format(i), img[i, :, :, :], global_step)
            writer.flush()
            if global_step % args.validate_every == 0 and global_step:
                ################################# Validation loop ##################################################
                model.eval()
                print('Computing errors for eval samples')
                metrics, val_si = validate(args, model, val_loader, criterion_d)
                print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel',
                                                                                             'log10',
                                                                                             'rms',
                                                                                             'sq_rel', 'log_rms',
                                                                                             'd1',
                                                                                             'd2',
                                                                                             'd3'))
                print("{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
                    metrics['silog'], metrics['abs_rel'], metrics['log_10'],
                    metrics['rmse'],
                    metrics['sq_rel'], metrics['rmse_log'], metrics['a1'],
                    metrics['a2'],
                    metrics['a3']), end='\n')

                for i in eval_metrics:
                    eval_summary_writer.add_scalar('eval_metrics/{}'.format(i), metrics[i], int(global_step))
                eval_summary_writer.flush()

                my_utils.save_checkpoint(model, optimizer, epoch, f"{args.name}_latest.pt",
                                         root=os.path.join(args.root, log_dir, "checkpoints"))
                # save best model
                # d1
                if metrics['a1'] > best_to_save['a1']:
                    my_utils.save_checkpoint(model, optimizer, epoch,
                                             f"epoch_{epoch}_global_step_{global_step}_best.pt",
                                             root=os.path.join(log_dir, f"checkpoints/global_step_{global_step}"))
                    best_to_save['a1'] = metrics['a1']
                    file = open(log_dir + f"/checkpoints/global_step_{global_step}/best_save_{global_step}.txt",
                                'w')
                    for k, v in sorted(metrics.items()):
                        file.write(str(k) + ': ' + str(v) + '\n')
                    file.close()
                    print('save best d1: {:.3f}'.format(metrics['a1']))
                # d2
                if metrics['a2'] > best_to_save['a2']:
                    my_utils.save_checkpoint(model, optimizer, epoch,
                                             f"epoch_{epoch}_global_step_{global_step}_best.pt",
                                             root=os.path.join(log_dir, f"checkpoints/global_step_{global_step}"))
                    best_to_save['a2'] = metrics['a2']
                    file = open(log_dir + f"/checkpoints/global_step_{global_step}/best_save_{global_step}.txt",
                                'w')
                    for k, v in sorted(metrics.items()):
                        file.write(str(k) + ': ' + str(v) + '\n')
                    file.close()
                    print('save best d2: {:.3f}'.format(metrics['a2']))
                # d3
                if metrics['a3'] > best_to_save['a3']:
                    my_utils.save_checkpoint(model, optimizer, epoch,
                                             f"epoch_{epoch}_global_step_{global_step}_best.pt",
                                             root=os.path.join(log_dir, f"checkpoints/global_step_{global_step}"))
                    best_to_save['a3'] = metrics['a3']
                    file = open(log_dir + f"/checkpoints/global_step_{global_step}/best_save_{global_step}.txt",
                                'w')
                    for k, v in sorted(metrics.items()):
                        file.write(str(k) + ': ' + str(v) + '\n')
                    file.close()
                    print('save best d3: {:.3f}'.format(metrics['a3']))

                for i in eval_metrics_low_better:
                    if metrics[i] < best_to_save[i]:
                        my_utils.save_checkpoint(model, optimizer, epoch,
                                                 f"epoch_{epoch}_global_step_{global_step}_best.pt",
                                                 root=os.path.join(log_dir,
                                                                   f"checkpoints/global_step_{global_step}"))
                        best_to_save[i] = metrics[i]
                        file = open(log_dir + f"/checkpoints/global_step_{global_step}/best_save_{global_step}.txt",
                                    'w')
                        for k, v in sorted(metrics.items()):
                            file.write(str(k) + ': ' + str(v) + '\n')
                        file.close()
                        print('save best {}: {:.3f}'.format(i, metrics[i]))

                model.train() 

        global_step += 1