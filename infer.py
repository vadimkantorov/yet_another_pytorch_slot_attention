import argparse
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import torch

import models
import clevr
import train

@torch.no_grad()
def main(args):
    frontend = models.ImagePreprocessor(resolution = args.resolution, crop = args.crop)
    model = models.SlotAttentionAutoEncoder(resolution = args.resolution, num_slots = args.num_slots, num_iterations = args.num_iterations, hidden_dim = args.hidden_dim).to(args.device)
        
    if args.checkpoint_tensorflow:
        model_state_dict = train.rename_and_transpose_tfcheckpoint(torch.load(args.checkpoint_tensorflow, map_location = 'cpu'))
        status = model.load_state_dict(model_state_dict, strict = False)
        assert set(status.missing_keys) == set(['encoder_pos.grid', 'decoder_pos.grid'])

    if args.checkpoint:
        model_state_dict = torch.load(args.checkpoint, map_location = 'cpu')['model_state_dict']
        status = model.load_state_dict(model_state_dict, strict = False)
        assert set(status.missing_keys) == set(['encoder_pos.grid', 'decoder_pos.grid'])

    test_set = clevr.CLEVR(args.dataset_root_dir, 'test')

    basename, image = test_set[2]
    image = image.unsqueeze(0).to(args.device)
    image = frontend(image)

    recon_combined, recons, masks, slots = model(image)

    recon_combined, recons, masks, image = map(lambda t: t.cpu().movedim(-3, -1).squeeze(0), [recon_combined, recons, masks, image])

    fig, ax = plt.subplots(1, args.num_slots + 2, figsize=(15, 2))
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[1].imshow(recon_combined)
    ax[1].set_title('Recon.')
    for i in range(args.num_slots):
      picture = recons[i] * masks[i] + (1 - masks[i])
      ax[i + 2].imshow(picture)
      ax[i + 2].set_title('Slot %s' % str(i + 1))
    for i in range(len(ax)):
      ax[i].grid(False)
      ax[i].axis('off')

    plt.savefig(args.savefig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for the model.')
    parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--hidden_dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--crop', type = int, nargs = 4, default = (29, 221, 64, 256))
    parser.add_argument('--resolution', type = int, nargs = 2, default = (128, 128))
    parser.add_argument('--dataset_root_dir', default = './CLEVR_v1.0')
    parser.add_argument('--device', default = 'cpu', choices = ['cuda', 'cpu'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint_tensorflow')
    parser.add_argument('--savefig', default = 'savefig.jpg')
    args = parser.parse_args()

    main(args)
