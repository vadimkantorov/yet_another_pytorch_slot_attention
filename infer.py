import os
import argparse
import matplotlib.pyplot as plt
import torch

import train


@torch.no_grad()
def main(args):
    dataset, collate_fn, batch_frontend = train.build_dataset(args)
    model = train.build_model(args)
    
    for b in args.triplets:
        output_path = args.savefig.format(b)
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        index = list(range(b * 3, (b + 1) * 3))

        images = torch.stack([dataset[idx][0] for idx in index])
        batch = batch_frontend(images.to(args.device))
        recon_combined, recons, masks, slots, attn = map(lambda t: t.cpu(), model(batch))
        
        renormalize = lambda x: x / 2. + 0.5 
        image, recon_combined, recons, masks, attn = [t.movedim(-3, -1) for t in [renormalize(batch).cpu(), renormalize(recon_combined), renormalize(recons), masks, attn]]

        fig, ax = plt.subplots(2 * len(index), 3 + args.num_slots, figsize=(15, 3 * len(index) ))
        for j in range(len(index)):
            ax[j * 2, 0].imshow(image[j])
            ax[j * 2, 0].set_title('Image')
            ax[j * 2, 1].imshow(recon_combined[j])
            ax[j * 2, 1].set_title('Recon.')
            
            for k, (name, masks_j) in enumerate(dict(masks = masks[j], attn = attn[j]).items()):
                K = j * 2 + k

                picture = recons[j] * masks_j + (1 - masks_j)
                entropy = -(masks_j * masks_j.clamp(min = 1e-12).log()).sum(dim = 0).squeeze(-1)
                
                ax[K, 2].imshow(entropy, cmap = 'jet')
                ax[K, 2].set_title(f'Entropy [{name}]')
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
    parser.add_argument('--num-iterations', default=3, type=int, help='Number of attention iterations.')
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
    parser.add_argument('--split-name', default = 'train', choices = ['train', 'val'])
    parser.add_argument('--coco-year', type = int, default = 2017)
    parser.add_argument('--coco-masks', action = 'store_true')
    parser.add_argument('--coco-mode', default = 'instances')
    args = parser.parse_args()

    main(args)
