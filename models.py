import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImagePreprocessor(nn.Module):
    def __init__(self, resolution, crop = tuple()):
        super().__init__()
        self.resolution = resolution
        self.crop = crop
        
    def forward(self, image):
        assert image.is_floating_point()
        image = image.sub_(0.5).mul_(2.0)

        if self.crop:
            image = image[..., self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]

        image = F.interpolate(image, self.resolution)
        image = image.clamp(-1, 1)
        return image

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
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size = kernel_size, stride = stride, padding = 1), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size = kernel_size, stride = stride, padding = 1), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size = kernel_size, stride = stride, padding = 1), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size = kernel_size, stride = stride, padding = 1), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size = kernel_size), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(hidden_dim, output_dim, kernel_size = output_kernel_size)
        )

class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_dim, resolution):
        super().__init__()
        self.dense = nn.Linear(4, hidden_dim)
        grid = torch.as_tensor(np.stack(np.meshgrid(*[np.linspace(0., 1., num=res) for res in resolution], sparse=False, indexing="ij"), axis=-1)).to(torch.float32)
        grid = grid.reshape(1, resolution[0], resolution[1], -1)
        grid = torch.cat([grid, 1 - grid], dim = -1)
        self.register_buffer('grid', grid)

    def forward(self, inputs):
        return inputs + self.dense(self.grid)


class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution = (128, 128), num_slots = 8, num_iterations = 3, decoder_initial_size = (8, 8), hidden_dim = 64):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
    
        self.encoder_cnn = SlotAttentionEncoder()
        
        self.decoder_initial_size = decoder_initial_size
        self.hidden_dim = hidden_dim

        self.decoder_cnn = SlotAttentionDecoder()

        self.encoder_pos = SoftPositionEmbed(self.hidden_dim, self.resolution)
        self.decoder_pos = SoftPositionEmbed(self.hidden_dim, self.decoder_initial_size)

        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.slot_attention = SlotAttention(
            num_iter = self.num_iterations,
            num_slots = self.num_slots,
            input_size = self.hidden_dim,
            slot_size = 64,
            mlp_hidden_size = 128)

    def forward(self, image):
        x = self.encoder_cnn(image)
        x = self.encoder_pos(x.movedim(1, -1))
        x = self.mlp(self.layer_norm(x))
        
        slots = self.slot_attention(x.flatten(start_dim = 1, end_dim = 2))
        x = slots.reshape(-1, 1, 1, slots.shape[-1]).expand(-1, *self.decoder_initial_size, -1)
        x = self.decoder_pos(x)
        x = self.decoder_cnn(x.movedim(-1, 1))

        x = F.interpolate(x, image.shape[-2:])

        recons, masks = x.unflatten(0, (image.shape[0], len(x) // image.shape[0])).split((image.shape[-3], 1), dim = 2)
        masks = masks.softmax(dim = 1)
        recon_combined = (recons * masks).sum(dim = 1)

        return recon_combined, recons, masks, slots

class SlotAttention(nn.Module):
    def __init__(self, num_iter, num_slots, input_size, slot_size, mlp_hidden_size, epsilon=1e-8):
        super().__init__()
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
        
        nn.init.xavier_uniform_(self.project_q.weight)
        nn.init.xavier_uniform_(self.project_k.weight)
        nn.init.xavier_uniform_(self.project_v.weight)
        
        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(inplace = True),
            nn.Linear(self.mlp_hidden_size, self.slot_size)
        )

    def forward(self, inputs):
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)
       
        slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(len(inputs), self.num_slots, self.slot_size, device = self.slots_mu.device)

        for _ in range(self.num_iter):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots)
            q *= self.slot_size ** -0.5
            
            attn_logits = torch.bmm(q, k.transpose(-1, -2))

            attn = F.softmax(attn_logits, dim = 1)
            attn = attn + self.epsilon

            bincount = attn.sum(dim = -1, keepdim = True)
            updates = torch.bmm(attn / bincount, v)
            
            slots = self.gru(updates.flatten(end_dim = 1), slots_prev.flatten(end_dim = 1)).reshape_as(slots)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots
