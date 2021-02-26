import os
import time
import datetime
import argparse
import functools
import torch
import torch.nn as nn

import models
import clevr
import torchvision

def main(args):
    os.makedirs(args.model_dir, exist_ok = True)

    train_set = clevr.CLEVR(args.dataset_root_dir, 'train', transform = torchvision.transforms.ToTensor(), filter = lambda scene_objects: len(scene_objects) <= 6)
    frontend = models.ImagePreprocessor(resolution = args.resolution, crop = args.crop)
    model = models.SlotAttentionAutoEncoder(resolution = args.resolution, num_slots = args.num_slots, num_iterations = args.num_iterations, hidden_dim = args.hidden_dim).to(args.device)
    criterion = nn.MSELoss()

    # model.load_state_dict(torch.load('./tmp/model6.ckpt')['model_state_dict'])


    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    start = time.time()
    iteration = 0
    for epoch in range(args.num_epochs):
        model.train()

        total_loss = 0

        for image_paths, images in train_dataloader:
            iteration += 1

            learning_rate = (args.learning_rate * (iteration / args.warmup_steps) if iteration < args.warmup_steps else args.learning_rate) * (args.decay_rate ** (iteration / args.decay_steps))

            optimizer.param_groups[0]['lr'] = learning_rate
            
            images = frontend(images.to(args.device))
            recon_combined, recons, masks, slots = model(images)
            loss = criterion(recon_combined, images)
            loss_item = float(loss)
            total_loss += loss_item

            del recons, masks, slots

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, iteration, loss_item)

        total_loss /= len(train_dataloader)

        print ("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss, datetime.timedelta(seconds=time.time() - start)))

        if not epoch % 10:
            torch.save(dict(model_state_dict = model.state_dict()), args.model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./checkpoints', type=str, help='where to save models' )
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for the model.')
    parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--num_train_steps', default=500000, type=int, help='Number of training steps.')
    parser.add_argument('--learning_rate', default=0.0004, type=float, help='Learning rate.')
    parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
    parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
    parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')

    parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
    parser.add_argument('--num_epochs', default=1000, type=int, help='number of workers for loading data')
    parser.add_argument('--hidden_dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--device', default = 'cuda', choices = ['cuda', 'cpu'])
    parser.add_argument('--resolution', type = int, nargs = 2, default = (128, 128))
    parser.add_argument('--crop', type = int, nargs = 4, default = (29, 221, 64, 256))
    parser.add_argument('--dataset_root_dir', default = './CLEVR_v1.0')
    args = parser.parse_args()

    main(args)
