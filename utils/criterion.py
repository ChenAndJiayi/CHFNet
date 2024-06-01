import torch
import torch.nn as nn
import numpy as np
class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.85):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return 10*loss

class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()

    
    def forward(self, depth, image):
        def gradient_x(img):
            gx = img[:,:,:-1,:] - img[:,:,1:,:]
            return gx

        def gradient_y(img):
            gy = img[:,:,:,:-1] - img[:,:,:,1:]
            return gy

        depth_grad_x = gradient_x(depth)
        depth_grad_y = gradient_y(depth)
        image_grad_x = gradient_x(image)
        image_grad_y = gradient_y(image)

        weights_x = torch.exp(-torch.mean(torch.abs(image_grad_x),1,True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_grad_y),1,True))
        smoothness_x = depth_grad_x*weights_x
        smoothness_y = depth_grad_y*weights_y

        loss_x = torch.mean(torch.abs(smoothness_x))
        loss_y = torch.mean(torch.abs(smoothness_y))

        loss = loss_x + loss_y
        
        return loss

class gradientLoss(nn.Module):
    def __init__(self):
        super(gradientLoss, self).__init__()
        self.edge_conv=nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        # edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_kx=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        edge_ky=np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k=np.stack((edge_kx, edge_ky))

        edge_k=torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight=nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad=False
    
    def forward(self, pred,depth):

        depth_grad=self.edge_conv(depth)
        depth_grad=depth_grad.contiguous().view(-1, 2, depth.size(2), depth.size(3))

        output_grad=self.edge_conv(pred)
        output_grad=output_grad.contiguous().view(-1, 2, pred.size(2), pred.size(3))

        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
        
        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
        
        loss = loss_dx + loss_dy
        
        return loss