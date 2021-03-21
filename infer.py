import argparse
import matplotlib.pyplot as plt
import torch

import models
import clevr
import train

def renormalize(x):
    """Renormalize from [-1, 1] to [0, 1]."""
    return x / 2. + 0.5

def get_prediction(model, batch, idx=0):
    recon_combined, recons, masks, slots, attn = map(lambda t: t.cpu(), model(batch))
    return [t.movedim(-3, -1) for t in [renormalize(batch)[idx].cpu(), renormalize(recon_combined)[idx], renormalize(recons)[idx], masks[idx], attn[idx]]]

@torch.no_grad()
def main(args):
    frontend, model = train.build_model(args)

    test_set = clevr.CLEVR(args.dataset_root_dir, args.split_name)

    image = test_set[args.index]['image']
    batch = frontend(image.unsqueeze(0).to(args.device))
    num_slots = args.num_slots

    # Predict.
    image, recon_combined, recons, masks, attn = get_prediction(model, batch)

    # Visualize.
    fig, ax = plt.subplots(2, 3 + num_slots, figsize=(15, 2))
    ax[0, 0].imshow(image)
    ax[0, 0].set_title('Image')
    ax[0, 1].imshow(recon_combined)
    ax[0, 1].set_title('Recon.')
    
    for k, (name, masks) in enumerate(dict(masks = masks, attn = attn).items()):
        picture = recons * masks + (1 - masks)
        entropy = -(masks * masks.clamp(min = 1e-12).log()).sum(dim = 0).squeeze(-1)
        ax[k, 2].imshow(entropy, cmap = 'jet')
        ax[k, 2].set_title(f'Entropy [{name}]')
        for i in range(num_slots):
            ax[k, i + 3].imshow(picture[i])
            ax[k, i + 3].set_title(f'Slot [{name}] {i + 1}')

        for a in ax[k]:
            a.grid(False)
            a.axis('off')

    plt.savefig(args.savefig)
    print(args.savefig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-slots', default=7, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num-iterations', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--hidden-dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--crop', type = int, nargs = 4, default = (29, 221, 64, 256))
    parser.add_argument('--resolution', type = int, nargs = 2, default = (128, 128))
    parser.add_argument('--dataset-root-dir', default = './CLEVR_v1.0')
    parser.add_argument('--device', default = 'cpu', choices = ['cuda', 'cpu'])
    parser.add_argument('--data-parallel', action = 'store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-tensorflow')
    parser.add_argument('--savefig', default = 'savefig.jpg')
    parser.add_argument('--index', type = int, default = 2)
    parser.add_argument('--split-name', default = 'val')
    args = parser.parse_args()

    main(args)
