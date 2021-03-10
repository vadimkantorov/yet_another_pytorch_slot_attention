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

    image = renormalize(batch)[idx]
    recon_combined = renormalize(recon_combined)[idx]
    recons = renormalize(recons)[idx]
    masks = masks[idx]
    return [t.movedim(-3, -1) for t in [image.cpu(), recon_combined, recons, masks, slots]]

@torch.no_grad()
def main(args):
    frontend, model = train.build_model(args)

    test_set = clevr.CLEVR(args.dataset_root_dir, args.split_name)

    image = test_set[args.index]['image']
    batch = frontend(image.unsqueeze(0).to(args.device))
    num_slots = args.num_slots

    # Predict.
    image, recon_combined, recons, masks, slots = get_prediction(model, batch)

    # Visualize.
    fig, ax = plt.subplots(1, num_slots + 2, figsize=(15, 2))
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[1].imshow(recon_combined)
    ax[1].set_title('Recon.')
    for i in range(num_slots):
      picture = recons[i] * masks[i] + (1 - masks[i])
      ax[i + 2].imshow(picture)
      ax[i + 2].set_title('Slot %s' % str(i + 1))
    for i in range(len(ax)):
      ax[i].grid(False)
      ax[i].axis('off')

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
