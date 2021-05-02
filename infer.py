import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import train
import eqv

@torch.no_grad()
def main(args):
    dataset, collate_fn, batch_frontend = train.build_dataset(args)
    model = train.build_model(args)
    
    criterion = eqv.EquivarianceLoss()
    aug = eqv.AffineTransform()
    transform_params = dict(angle = 20.0)
        
    renormalize = lambda x: x / 2. + 0.5 
    
    train.set_seed(args.seed)

    for b in args.triplets:
        output_path = args.savefig.format(b)
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        index = list(range(b * 3, (b + 1) * 3))

        images = torch.stack([dataset[idx][0] for idx in index])
        batch = batch_frontend(images.to(args.device))

        slots_init = None #model.slot_attention.slots_mu.expand(len(batch), model.num_slots, -1)
        
        #train.set_seed(args.seed)
        recon_combined, recons, masks, slots, attn = map(lambda t: t.cpu().movedim(-3, -1), model(batch, slots = slots_init))
        recon_combined, recons, image = renormalize(recon_combined), renormalize(recons), renormalize(batch).cpu().movedim(-3, -1)
        
        batch_aug = aug(batch, **transform_params)
        
        #train.set_seed(args.seed)
        recon_combined_aug, recons_aug, masks_aug, slots_aug, attn_aug = map(lambda t: t.cpu().movedim(-3, -1), model(batch_aug, slots = slots_init))
        recon_combined_aug, recons_aug, image_aug = renormalize(recon_combined_aug), renormalize(recons_aug), renormalize(batch_aug).cpu().movedim(-3, -1)
        
        loss_aug, masks_aug_, recons_aug_ = criterion( masks_aug.squeeze(-1), aug(masks.squeeze(-1), **transform_params), recons_aug )
        
        print(loss_aug)

        fig, ax = plt.subplots(2 * len(index), 3 + args.num_slots, figsize=(15, 3 * len(index) ))
        for j in range(len(index)):
            for k, (name, (image_j, recons_j, recon_combined_j, masks_j)) in enumerate(dict(orig = (image[j], recons[j], recon_combined[j], masks[j]), aug = (image_aug[j], recons_aug_[j], recon_combined_aug[j], masks_aug_[j])).items()):
                K = j * 2 + k
            
                ax[K, 0].imshow(image_j)
                ax[K, 0].set_title(f'Image [{name}]')
                ax[K, 1].imshow(recon_combined_j)
                ax[K, 1].set_title(f'Recon. [{name}]')

                entropy = -(masks_j * masks_j.clamp(min = 1e-12).log()).sum(dim = 0, keepdim = True)
                #entropy = (entropy < 1.0).float()
                
                #picture = recons_j * masks_j + (1 - masks_j)
                picture = masks_j #(masks_j > 0.7).float() * entropy

                im = ax[K, 2].imshow(entropy.squeeze(0).squeeze(-1), cmap = 'jet')
                ax[K, 2].set_title(f'Entropy [{name}]')
                plt.colorbar(im, ax = ax[K, 2])
                for i in range(args.num_slots):
                    ax[K, i + 3].imshow(picture[i])
                    ax[K, i + 3].set_title(f'Slot [{name}] {i + 1}')
                for a in ax[K]:
                    a.grid(False)
                    a.axis('off')

        plt.subplots_adjust(left = 0, right = 1, bottom = 0, top = 0.97)
        plt.savefig(output_path)
        print(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-slots', default=7, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num-iter', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--hidden-dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--crop', type = int, nargs = 4, default = (29, 221, 64, 256))
    parser.add_argument('--resolution', type = int, nargs = 2, default = (128, 128))
    parser.add_argument('--device', default = 'cpu', choices = ['cuda', 'cpu'])
    parser.add_argument('--data-parallel', action = 'store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-tensorflow')
    parser.add_argument('--savefig', default = 'savefig/savefig_{}.jpg')
    parser.add_argument('--triplets', type = int, nargs = '*', default = [0])
    parser.add_argument('--dataset', default = 'CLEVR', choices = ['CLEVR', 'COCO'])
    parser.add_argument('--dataset-root-dir', default = './CLEVR_v1.0')
    parser.add_argument('--dataset-split-name', default = 'train', choices = ['train', 'val'])
    parser.add_argument('--coco-year', type = int, default = 2017)
    parser.add_argument('--coco-masks', action = 'store_true')
    parser.add_argument('--coco-mode', default = 'instances')
    parser.add_argument('--seed', type = int, default = 0)
    args = parser.parse_args()

    main(args)
