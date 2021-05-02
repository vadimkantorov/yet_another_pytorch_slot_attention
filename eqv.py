import numpy as np
import scipy.optimize
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Tfunc
import kornia as K

def entropy(tensor, dim = 1, eps = 1e-12, normalized = False, keepdim = False):
    return -(torch.mean if normalized else torch.sum)(tensor * (tensor + eps).log(), dim = dim, keepdim = keepdim)

def mask_topk(x, K = 1, dim = -1):
    return torch.zeros_like(x).scatter_(dim, x.topk(max(1, min(K, x.size(dim))), dim = dim)[1], 1.0)


class AffineTransform(nn.Module):
    # https://github.com/monniert/dti-clustering/blob/b57a77d4c248b16b4b15d6509b6ec493c53257ef/src/model/transformer.py#L281
    # https://github.com/kornia/kornia/blob/3606cf9c3d1eb3aabd65ca36a0e7cb98944c01ba/kornia/geometry/transform/imgwarp.py#L122
    def forward(self, x : 'NCHW', angle = 0.0):
        #center = torch.tensor([img.shape[-1] / 2, img.shape[-2] / 2]).unsqueeze(0)
        #angle = torch.tensor(angle).unsqueeze(0)
        #scale = torch.ones(2).unsqueeze(0)
        #R = K.get_rotation_matrix2d(center, angle, scale)
        #return K.warp_affine(img.float(), R, dsize = img.shape[-2:]).to(torch.uint8)
        return Tfunc.rotate(x, angle)

def mask_dice_loss(pred, true):
    # https://github.com/facebookresearch/detr/blob/master/models/segmentation.py
    # https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/dice.html#dice_loss
    pred, true = pred.flatten(start_dim = -2).float(), true.flatten(start_dim = -2).float()
    numerator = 2 * (pred * true).float().sum(-1)
    denominator = (pred + true).sum(-1)
    return 1 - (numerator + 1) / (denominator + 1)

def mask_dice_loss_clipped(pred, true, entropy_threshold = 1.0):
    pred, true = pred.flatten(start_dim = -2).float(), true.flatten(start_dim = -2).float()

    mask_pred, mask_true = (entropy(pred, dim = 1, keepdim = True) < entropy_threshold, entropy(true, dim = 1, keepdim = True) < entropy_threshold)
    mask_pred, mask_true = mask_pred * mask_topk(pred, dim = 1), mask_true * mask_topk(true, dim = 1)
   
    numerator = 2 * (pred * mask_pred * true * mask_true ).float().sum(-1)
    denominator = (pred * mask_pred + true * mask_true ).sum(-1)
    
    return 1 - (numerator + 1) / (denominator + 1)

def mask_cross_entropy(pred, true):
    return F.binary_cross_entropy(pred, true, reduction = 'none').mean(dim = (-2, -1))

def mask_kl_div(pred, true, eps = 1e-15):
    return true.mul((true + eps).log() - (pred + eps).log()).mean(dim = (-2, -1))
    #return F.kl_div(pred.log(), true, reduction = 'none').mean(dim = (-2, -1))

class EquivarianceLoss(nn.Module):
    def __init__(self, mode = ''):
        super().__init__()
    
    def linear_row2col_assignment(self, C):
        return torch.stack([torch.as_tensor(c[r.argsort()]) for r, c in map(scipy.optimize.linear_sum_assignment, C.detach().cpu())]).to(C.device)

    def mask_divergence(self, pred, true):
        return mask_kl_div(pred, true)

    def compute_matching_cost_matrix(self, outputs, targets):
        return self.mask_divergence(*torch.broadcast_tensors(outputs.unsqueeze(2), targets.unsqueeze(1).detach()))
    
    def permute_rows(self, outputs, R2C):
        row_perm = R2C.argsort(dim = 1)
        return outputs.gather(1, row_perm[(...,) + (None,) * (outputs.ndim - row_perm.ndim)].expand_as(outputs))
        
    def forward(self, pred_masks : 'BKHW', aug_masks: 'BKHW', *args):
        # TODO: feature loss?
        cost_matrix : 'BKK' = self.compute_matching_cost_matrix(pred_masks, aug_masks)
        R2C : 'BK' = self.linear_row2col_assignment(cost_matrix)
        pred_masks_permuted = self.permute_rows(pred_masks, R2C)

        #l = mask_dice_loss_clipped(pred_masks_permuted, aug_masks.detach())
        l = self.mask_divergence(pred_masks_permuted, aug_masks.detach())
        return (l.mean(), pred_masks_permuted.unsqueeze(-1)) + tuple(self.permute_rows(t, R2C)  for t in args)

if __name__ == '__main__':
    img_path = 'CLEVR_with_masks/images/CLEVR6val/CLEVR_CLEVR6val_070019.png'
    
    model = nn.Identity()
    criterion = EquivarianceLoss()
    transform = Rotation()
    
    batch = torch.as_tensor(cv2.imread(img_path)).movedim(-1, 0).flip(0).unsqueeze(0) / 255.0

    masks = model(batch)
    masks_aug = model(transform(batch))

    loss = criterion(transform(masks), masks_aug)
    print(loss)

    #outputs = torch.rand(4, 8, 16, 16)
    #targets = torch.rand(4, 8, 16, 16)
    #print(m(outputs, targets))
