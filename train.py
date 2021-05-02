import os
import sys
import time
import datetime
import argparse
import functools
import torch
import torch.nn as nn

import clevr
import coco
import models
import eqv

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_model(args):
    model = models.SlotAttentionAutoEncoder(resolution = args.resolution, num_slots = args.num_slots, num_iter = args.num_iter, hidden_dim = args.hidden_dim).to(args.device)
        
    if args.checkpoint or args.checkpoint_tensorflow:
        model_state_dict = torch.load(args.checkpoint, map_location = 'cpu')['model_state_dict'] if args.checkpoint else rename_and_transpose_tfcheckpoint(torch.load(args.checkpoint_tensorflow, map_location = 'cpu')) 
        model_state_dict = {'.'.join(k.split('.')[1:]) if k.startswith('module.') else k : v for k, v in model_state_dict.items()}
        status = model.load_state_dict(model_state_dict, strict = False)
        assert not status.missing_keys or set(status.missing_keys) == set(['encoder_pos.grid', 'decoder_pos.grid'])
    
    if args.data_parallel:
        model = nn.DataParallel(model)
    model = model.to(args.device).eval()

    return model

def build_dataset(args, filter = None):
    assert os.path.exists(args.dataset_root_dir), f'provided dataset path [{args.dataset_root_dir}] does not exist'
    
    if args.dataset == 'CLEVR':
        dataset = clevr.CLEVR(args.dataset_root_dir, args.dataset_split_name, filter = filter)
        batch_frontend = models.ClevrImagePreprocessor(resolution = args.resolution, crop = args.crop)
        collate_fn = lambda batch: (torch.utils.data.dataloader.default_collate([t[0] for t in batch]), [t[1] for t in batch])

    elif args.dataset == 'COCO':
        PATHS = dict(
            train = (os.path.join(args.dataset_root_dir, f'train{args.coco_year}'), os.path.join(args.dataset_root_dir, 'annotations', f'{args.coco_mode}_train{args.coco_year}.json')),
            val = (os.path.join(args.dataset_root_dir, f'val{args.coco_year}'), os.path.join(args.dataset_root_dir, 'annotations', f'{args.coco_mode}_val{args.coco_year}.json')),
        )
        dataset = coco.CocoDetection(*PATHS[args.dataset_split_name], transforms = models.CocoImagePreprocessorSimple(args.dataset_split_name), return_masks = args.coco_masks)
        batch_frontend = nn.Identity()
        collate_fn = lambda batch: (torch.utils.data.dataloader.default_collate([t[0] for t in batch]), [t[1] for t in batch])

    return dataset, collate_fn, batch_frontend

def build_criterion(args):
    reconstruction = nn.MSELoss()
    equivariance = eqv.EquivarianceLoss()

    if args.loss_scale_equivariance == 0 and args.loss_scale_reconstruction > 0:
        return lambda true_images, pred_images, **kwargs: dict(reconstruction = reconstruction(pred_images, true_images), equivariance = 0)

    return lambda true_images, pred_images, aug_pred_masks, pred_aug_masks: dict(reconstruction = reconstruction(pred_images, true_images), equivariance = equivariance(pred_aug_masks, aug_pred_masks)[0] )

def sample_affine_transform_params(batch_size = 1, angle_min = -20, angle_max = 20, generator = None):
    return dict(angle = angle_min + float(torch.rand(batch_size, generator = generator)) * (angle_max - angle_min))

def mix_losses(args, losses):
    return sum(v * getattr(args, 'loss_scale_' + k) for k, v in losses.items())

def main(args):
    set_seed(args.seed)

    log = open(args.log, 'w')
    
    os.makedirs(args.checkpoint_dir, exist_ok = True)
 
    dataset, collate_fn, batch_frontend = build_dataset(args, filter = lambda scene_objects: len(scene_objects) <= 6)
    
    model = build_model(args)
    criterion = build_criterion(args)
    aug = eqv.AffineTransform()

    data_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, num_workers = args.num_workers, collate_fn = collate_fn, shuffle = True)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

    start = time.time()
    iteration = 0
    for epoch in range(args.num_epochs):
        model.train()

        total_loss = 0
        total_losses = dict()

        for i, (images, extra) in enumerate(data_loader):
            learning_rate = (args.learning_rate * (iteration / args.warmup_steps) if iteration < args.warmup_steps else args.learning_rate) * (args.decay_rate ** (iteration / args.decay_steps))
            optimizer.param_groups[0]['lr'] = learning_rate
           
            transform_params = sample_affine_transform_params()

            true_images = batch_frontend(images.to(args.device))
            aug_images = aug(true_images, **transform_params)

            pred_images, _, pred_masks,     _, _ = model(true_images)
            _, _,           pred_aug_masks, _, _ = model(aug_images)
           
            aug_pred_masks = aug(pred_masks.squeeze(-3), **transform_params)
            
            losses = criterion(pred_images, true_images, aug_pred_masks = aug_pred_masks, pred_aug_masks = pred_aug_masks.squeeze(-3))
            loss = mix_losses(args, losses)
            
            print({k: f'{v:.04f}' for k, v in losses.items()})
            loss_item = float(loss)
            total_loss += loss_item
            total_losses = {k : float(v) + total_losses.get(k, 0.0) for k, v in losses.items()}

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch:', epoch, '#', iteration, '|', i, '/', len(data_loader), 'Loss:', loss_item)
            iteration += 1

        total_loss /= len(data_loader)
        total_losses = {k : v / len(data_loader) for k, v in total_losses.items()}

        for f in [sys.stdout, log]:
            print('Epoch:', epoch, 'Loss:', total_loss, 'Time:', datetime.timedelta(seconds = time.time() - start), 'All losses:', total_losses, file = f, flush = True)

        if epoch % args.checkpoint_epoch_interval == 0 or epoch == args.num_epochs - 1:
            model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(dict(model_state_dict = model_state_dict), os.path.join(args.checkpoint_dir, args.checkpoint_pattern.format(epoch = epoch)))

def rename_and_transpose_tfcheckpoint(ckpt):
    # converted with https://github.com/vadimkantorov/tfcheckpoint2pytorch
    # checkpoint format: https://www.tensorflow.org/guide/checkpoint
   
    replace = {
        'network/layer_with_weights-0/': '',
        '/.ATTRIBUTES/VARIABLE_VALUE': '',
        '/': '_',
        'encoder_cnn_layer_with_weights-': 'encoder_cnn.',
        'decoder_cnn_layer_with_weights-': 'decoder_cnn.',
        'slot_attention_': 'slot_attention.',
        'mlp_layer_with_weights-': 'mlp.',
        'encoder_pos_': 'encoder_pos.',
        'decoder_pos_': 'decoder_pos.',
        '_kernel': '.weight',
        '_bias': '.bias',
        '_gamma': '.weight',
        '_beta': '.bias',
        'slot_attention.gru.weight': 'slot_attention.gru.weight_ih',
        'slot_attention.gru_recurrent.weight': 'slot_attention.gru.weight_hh'
    }
    
    ckpt = {functools.reduce(lambda acc, from_to: acc.replace(*from_to), replace.items(), k) : v for k, v in ckpt.items() if k.startswith('network/layer_with_weights-0/') and k.endswith('.ATTRIBUTES/VARIABLE_VALUE') and '.OPTIMIZER_SLOT' not in k}
    ckpt = { ('.'.join(k.split('.')[:-2] + [str(int(k.split('.')[-2]) * 2), k.split('.')[-1]]) if 'encoder_cnn.' in k or 'decoder_cnn.' in k or k.startswith('mlp.') or 'slot_attention.mlp.' in k else k) : v for k, v in ckpt.items() }
    ckpt['slot_attention.gru.bias_ih'], ckpt['slot_attention.gru.bias_hh'] = ckpt.pop('slot_attention.gru.bias').unbind()
    return {k : v.permute(3, 2, 0, 1) if v.ndim == 4 else v.t() if v.ndim == 2 else v for k, v in ckpt.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--num-epochs', default=10000, type=int, help='number of workers for loading data')
    parser.add_argument('--num-train-steps', default=500000, type=int, help='Number of training steps.')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size for the model.')
    parser.add_argument('--data-parallel', action = 'store_true') 
    parser.add_argument('--device', default = 'cuda', choices = ['cuda', 'cpu'])
    parser.add_argument('--num-workers', default=16, type=int, help='number of workers for loading data')
    parser.add_argument('--learning-rate', default=0.0004, type=float, help='Learning rate.')
    parser.add_argument('--warmup-steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
    parser.add_argument('--decay-rate', default=0.5, type=float, help='Rate for the learning rate decay.')
    parser.add_argument('--decay-steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    
    parser.add_argument('--num-iter', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--num-slots', default=7, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--hidden-dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--resolution', type = int, nargs = 2, default = (128, 128))
    parser.add_argument('--crop', type = int, nargs = 4, default = (29, 221, 64, 256))
    parser.add_argument('--loss-scale-reconstruction', type = float, default = 1.0)
    parser.add_argument('--loss-scale-equivariance', type = float, default = 0.5)
   
    parser.add_argument('--checkpoint-dir', default='./checkpoints', type=str, help='where to save models' )
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-epoch-interval', type = int, default = 2)
    parser.add_argument('--checkpoint-pattern', default = 'ckpt_{epoch:04d}.pt')
    parser.add_argument('--checkpoint-tensorflow')
    
    parser.add_argument('--dataset', default = 'CLEVR', choices = ['CLEVR', 'COCO'])
    parser.add_argument('--dataset-root-dir', default = './CLEVR_v1.0')
    parser.add_argument('--dataset-split-name', default = 'train', choices = ['train', 'val'])
    parser.add_argument('--coco-year', type = int, default = 2017)
    parser.add_argument('--coco-masks', action = 'store_true')
    parser.add_argument('--coco-mode', default = 'instances')

    parser.add_argument('--log', default = 'log.txt')
    
    main(parser.parse_args())
