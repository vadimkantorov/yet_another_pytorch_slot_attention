import os
import json
import torch
import torchvision
import numpy as np

class CLEVR(torchvision.datasets.VisionDataset):
    def __init__(self, root, split_name, transform = torchvision.transforms.ToTensor(), loader = torchvision.datasets.folder.default_loader, filter = None):
        super().__init__(root, transform = transform)
        self.loader = loader
        scenes_json_path = os.path.join(root, 'scenes', f'CLEVR_{split_name}_scenes.json')
        images_split_dir = os.path.join(root, 'images', split_name)
        
        self.image_paths = sorted(os.path.join(images_split_dir, basename) for basename in os.listdir(images_split_dir))
        self.metadata = {s['image_filename'] : s['objects'] for s in json.load(open(scenes_json_path))['scenes'] } if os.path.exists(scenes_json_path) else { os.path.basename(image_path) : {} for image_path in self.image_paths}
    
        if filter != None:
            self.image_paths = [image_path for image_path in self.image_paths if filter(self.metadata[os.path.basename(image_path)])]
            self.metadata = {os.path.basename(image_path) : self.metadata[os.path.basename(image_path)] for image_path in self.image_paths}

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = image_path.replace('images', 'masks').replace('.png', '.npy')
        
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)

        if os.path.exists(mask_path):
            mask = torch.as_tensor(np.load(mask_path))
        else:
            mask = None
        
        return image, dict(mask = mask, image_name = os.path.basename(image_path))
