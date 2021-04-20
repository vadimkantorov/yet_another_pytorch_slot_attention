import numpy as np
import scipy.optimize
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms.functional as Tfunc
import kornia as K

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

def dice_loss(inputs : 'probs', targets : 'probs'):
    # https://github.com/facebookresearch/detr/blob/master/models/segmentation.py
    # https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/dice.html#dice_loss
    inputs, targets = inputs.flatten(start_dim = -2).float(), targets.flatten(start_dim = -2).float()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = (inputs + targets).sum(-1)
    return 1 - (numerator + 1) / (denominator + 1)

class EquivarianceLoss(nn.Module):
    def linear_row2col_assignment(self, C):
        return torch.stack([torch.as_tensor(c[r.argsort()]) for r, c in map(scipy.optimize.linear_sum_assignment, C.cpu())]).to(C.device)

    def compute_matching_cost_matrix(self, outputs, targets):
        return dice_loss(outputs.unsqueeze(2), targets.unsqueeze(1))
    
    def permute_rows(self, outputs, R2C):
        row_perm = R2C.argsort(dim = 1)
        return outputs.gather(1, row_perm[..., None, None].expand(-1, -1, *outputs.shape[-2:]))
        
    def forward(self, aug_masks: 'BKHW', pred_masks : 'BKHW'):
        # TODO: feature loss?
        cost_matrix : 'BKK' = self.compute_matching_cost_matrix(pred_masks, aug_masks)
        R2C : 'BK' = self.linear_row2col_assignment(cost_matrix)
        
        return dice_loss(self.permute_rows(aug_masks, R2C), pred_masks).sum()

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
