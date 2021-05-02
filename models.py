import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import transforms as T
import torchvision

class ClevrImagePreprocessor(nn.Module):
    def __init__(self, resolution, crop = tuple(), rgb_mean = 0.5, rgb_std = 0.5):
        super().__init__()
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.resolution = resolution
        self.crop = crop
        
    def forward(self, img, normalize = True, interpolate_mode = 'bilinear'):
        assert img.is_floating_point()
        img = (img - self.rgb_mean) / self.rgb_std if normalize else img

        img = img[..., self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]] if self.crop else img

        img = F.interpolate(img, self.resolution, mode = interpolate_mode)
        img = img.clamp(-1 if normalize else 0, 1)

        return img

class CocoImagePreprocessorSimple(nn.Module):
    def __init__(self, split_name, rgb_mean = [0.485, 0.456, 0.406], rgb_std = [0.229, 0.224, 0.225], resolution = (128, 128)):
        super().__init__()
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.resolution = resolution
    
    def forward(self, img, targets, normalize = True, interpolate_mode = 'bilinear'):
        img = torchvision.transforms.functional.to_tensor(img)
        img = torchvision.transforms.functional.normalize(img, self.rgb_mean, self.rgb_std) if normalize else img
        
        img = F.interpolate(img, self.resolution, mode = interpolate_mode) if img.ndim == 4 else F.interpolate(img.unsqueeze(0), self.resolution, mode = interpolate_mode).squeeze(0)
        img = img.clamp(-1 if normalize else 0, 1)

        return img, targets

class CocoImagePreprocessor(nn.Sequential):
    def __init__(self, split_name, rgb_mean = [0.485, 0.456, 0.406], rgb_std = [0.229, 0.224, 0.225], scales_train_selectA = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800], scales_train_selectB = [400, 500, 600], resolution_val = [800], max_size = 1333, crop_train = (384, 600)):
        
        if split_name == 'train':
            super().__init__(
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales_train_selectA, max_size = max_size),
                    T.Compose([
                        T.RandomResize(scales_train_selectB),
                        T.RandomSizeCrop(*crop_train),
                        T.RandomResize(scales, max_size = max_size),
                    ])
                ),
                T.ToTensor(),
                T.Normalize(rgb_mean, rgb_std)
            )

        elif split_name == 'val':
            super().__init__(
                T.RandomResize(resolution_val , max_size = max_size),
                T.ToTensor(),
                T.Normalize(rgb_mean, rgb_std)
            )
    
    def forward(self, *inputs):
        for module in self:
            inputs = module(*inputs)
        return inputs

class SlotAttention(nn.Module):
    def __init__(self, num_iter, num_slots, input_size, slot_size, mlp_hidden_size, epsilon=1e-8, simple = False, project_inputs = False, gain = 1, temperature_factor = 1):
        super().__init__()
        self.temperature_factor = temperature_factor
        self.num_iter = num_iter
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.input_size = input_size

        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots  = nn.LayerNorm(slot_size)
        self.norm_mlp    = nn.LayerNorm(slot_size)

        self.slots_mu        = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, 1, self.slot_size)))
        self.slots_log_sigma = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, 1, self.slot_size)))
        
        self.project_q = nn.Linear(slot_size, slot_size, bias = False)
        self.project_k = nn.Linear(input_size, slot_size, bias = False)
        self.project_v = nn.Linear(input_size, slot_size, bias = False)
        
        nn.init.xavier_uniform_(self.project_q.weight, gain = gain)
        nn.init.xavier_uniform_(self.project_k.weight, gain = gain)
        nn.init.xavier_uniform_(self.project_v.weight, gain = gain)
        
        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(inplace = True),
            nn.Linear(self.mlp_hidden_size, self.slot_size)
        )

        self.simple = simple
        if self.simple:
            assert slot_size == input_size
            self.norm_mlp = nn.Identity()
            del self.gru; self.gru = lambda x, h, alpha = 0.5: h * alpha + x * (1 - alpha)
            del self.mlp; self.mlp = torch.zeros_like
        self.project_x = nn.Linear(input_size, input_size) if project_inputs else nn.Identity()

    def forward(self, inputs : 'BTC', num_iter = 0, slots : 'BSC' = None) -> '(BSC, BST, BST)':
        inputs = self.project_x(inputs)

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)
       
        if slots is None:
            slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(len(inputs), self.num_slots, self.slot_size, device = self.slots_mu.device)

        for _ in range(num_iter or self.num_iter):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots)
            q *= self.slot_size ** -0.5
            
            attn_logits = torch.bmm(q, k.transpose(-1, -2))

            attn_pixelwise = F.softmax(attn_logits / self.temperature_factor, dim = 1)
            attn_slotwise = F.normalize(attn_pixelwise + self.epsilon, p = 1, dim = -1)

            updates = torch.bmm(attn_slotwise, v)
            
            slots = self.gru(updates.flatten(end_dim = 1), slots_prev.flatten(end_dim = 1)).reshape_as(slots)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_logits, attn_slotwise

class SlotAttentionEncoder(nn.Sequential):
    def __init__(self, hidden_dim = 64, kernel_size = 5, padding = 2):
        super().__init__(
            nn.Conv2d(3,          hidden_dim, kernel_size = kernel_size, padding = padding), nn.ReLU(inplace = True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = kernel_size, padding = padding), nn.ReLU(inplace = True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = kernel_size, padding = padding), nn.ReLU(inplace = True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = kernel_size, padding = padding), nn.ReLU(inplace = True),
        )

class SlotAttentionDecoder(nn.Sequential):
    def __init__(self, hidden_dim = 64, output_dim = 4, kernel_size = 5, padding = 2, stride = 2, output_kernel_size = 3, output_padding = 1):
        super().__init__(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size = kernel_size, stride = stride, padding = 1), nn.ReLU(inplace = True), # 3
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size = kernel_size, stride = stride, padding = 1), nn.ReLU(inplace = True), # 2
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size = kernel_size, stride = stride, padding = 1), nn.ReLU(inplace = True), # 2
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size = kernel_size, stride = stride, padding = 1), nn.ReLU(inplace = True), # 3
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size = kernel_size), nn.ReLU(inplace = True), # 1
            nn.ConvTranspose2d(hidden_dim, output_dim, kernel_size = output_kernel_size) # 1
        )


class PositionEmbeddingImplicit(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(4, hidden_dim)

    def forward(self, x):
        spatial_shape = x.shape[-3:-1]
        grid = torch.stack(torch.meshgrid(*[torch.linspace(0., 1., r, device = x.device) for r in spatial_shape]), dim = -1)
        grid = torch.cat([grid, 1 - grid], dim = -1)
        return x + self.dense(grid)

class PositionEmbeddingSine(nn.Module):
    # https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        not_mask = torch.ones_like(x)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return x + pos


class PositionEmbeddingLearned(nn.Module):
    # https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        #TODO: assert that x.shape matches the passed row_embed, col_embed
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return x + pos


class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution = (128, 128), num_slots = 8, num_iter = 3, decoder_initial_size = (8, 8), hidden_dim = 64, interpolate_mode = 'bilinear', position_encoding_layer = PositionEmbeddingImplicit):
        super().__init__()
        self.interpolate_mode = interpolate_mode
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iter = num_iter
        self.decoder_initial_size = decoder_initial_size
        self.hidden_dim = hidden_dim
        
        self.encoder_cnn = SlotAttentionEncoder(hidden_dim = self.hidden_dim)
        self.encoder_pos = position_encoding_layer(self.hidden_dim)
        
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.slot_attention = SlotAttention(
            num_iter = self.num_iter,
            num_slots = self.num_slots,
            input_size = self.hidden_dim,
            slot_size = self.hidden_dim,
            mlp_hidden_size = 128)
        
        self.decoder_pos = position_encoding_layer(self.hidden_dim)
        self.decoder_cnn = SlotAttentionDecoder(hidden_dim = self.hidden_dim, output_dim = 4)

    def forward(self, image, slots = None):
        x = self.encoder_cnn(image).movedim(1, -1)
        x = self.encoder_pos(x)
        x = self.mlp(self.layer_norm(x))
        
        slots, attn_logits, attn_slotwise = self.slot_attention(x.flatten(start_dim = 1, end_dim = 2), slots = slots)
        x = slots.reshape(-1, 1, 1, slots.shape[-1]).expand(-1, *self.decoder_initial_size, -1)
        x = self.decoder_pos(x)
        x = self.decoder_cnn(x.movedim(-1, 1))
        
        x = F.interpolate(x, image.shape[-2:], mode = self.interpolate_mode)

        x = x.unflatten(0, (len(image), len(x) // len(image)))

        recons, masks = x.split((3, 1), dim = 2)
        masks = masks.softmax(dim = 1)
        recon_combined = (recons * masks).sum(dim = 1)

        return recon_combined, recons, masks, slots, attn_slotwise.unsqueeze(-2).unflatten(-1, x.shape[-2:])
