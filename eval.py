import argparse
import matplotlib.pyplot as plt
import torch

import models
import clevr
import train
import infer
import metrics

@torch.no_grad()
def main(args):
    frontend, model = train.build_model(args)

    test_set = clevr.CLEVR(args.dataset_root_dir, 'train', filter = lambda scene_objects: len(scene_objects) - sum(o['padding'] for o in scene_objects) <= 6)
    
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ari = []
    for i, batch in enumerate(test_dataloader):
        images, masks = map(batch.get, ['image', 'mask'])
        images = frontend(images.to(args.device))
        masks = frontend(masks.squeeze(-1).to(args.device).div(255.0), bipole = False)
        
        mask_true = masks.flatten(start_dim = 2).transpose(-1, -2)
        recon_combined, recons, masks, slots = model(images)
        mask_pred = masks.flatten(start_dim = 2).transpose(-1, -2)

        ari.extend(metrics.adjusted_rand_index(mask_true, mask_pred).tolist())

    print('Num examples:', len(test_set), 'Adjusted Rand Index:', float(torch.tensor(ari).mean()))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size for the model.')
    parser.add_argument('--num-slots', default=7, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num-iterations', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--hidden-dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--crop', type = int, nargs = 4, default = (29, 221, 64, 256))
    parser.add_argument('--resolution', type = int, nargs = 2, default = (128, 128))
    parser.add_argument('--dataset-root-dir', default = './CLEVR_with_masks')
    parser.add_argument('--device', default = 'cpu', choices = ['cuda', 'cpu'])
    parser.add_argument('--num-workers', default=16, type=int, help='number of workers for loading data')
    parser.add_argument('--data-parallel', action = 'store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-tensorflow')
    args = parser.parse_args()

    main(args)
