import torch
import my_utils
from tqdm import tqdm
import numpy as np

def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 
            'log10':log10.item(), 'silog':silog.item()}


def cropping_img(args, pred, gt_depth):
    min_depth_eval = args.min_depth_eval

    max_depth_eval = args.max_depth_eval
    
    pred[torch.isinf(pred)] = min_depth_eval
    pred[torch.isnan(pred)] = min_depth_eval
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = min_depth_eval

    valid_mask = torch.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    if args.dataset == 'kitti':
        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            gt_depth = gt_depth[top_margin:top_margin +
                            352, left_margin:left_margin + 1216]            

        if args.kitti_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = torch.zeros(valid_mask.shape).to(
                device=valid_mask.device)

            if args.kitti_crop == 'garg_crop':
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                          int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.kitti_crop == 'eigen_crop':
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
            else:
                eval_mask = valid_mask

    elif args.dataset == 'nyudepthv2':
        eval_mask = torch.zeros(valid_mask.shape).to(device=valid_mask.device)
        eval_mask[45:471, 41:601] = 1
    else:
        eval_mask = valid_mask

    valid_mask = torch.logical_and(valid_mask, eval_mask)

    return pred[valid_mask], gt_depth[valid_mask]

def my_validate(args, model, test_loader):
    with torch.no_grad():
        # val_bins = RunningAverage()
        metrics = my_utils.RunningAverageDict()
        
        for  batch in tqdm(test_loader):
            img = batch['image'].cuda() #####
            depth = batch['depth'].cuda() ##############
            #####################3
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            depth= depth.unsqueeze(0)

            pred, _, _, _,_  = model(img)
            pred = pred.squeeze().cpu().numpy()
            """ pred中有一些奇异值, 将其统一 """
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