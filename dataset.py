import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from ccs import shape_targets


def paired_files(image_root, gt_root):
    extensions = ('.jpg', '.jpeg', '.png')
    def index(root):
        files = {}
        for name in os.listdir(root):
            if name.lower().endswith(extensions):
                stem = os.path.splitext(name)[0]
                if stem in files:
                    raise ValueError(f"Duplicate sample stem: {stem} in {root}")
                files[stem] = os.path.join(root, name)
        return files
    images, masks = index(image_root), index(gt_root)
    if set(images) != set(masks) or not images:
        raise ValueError("Image/mask filenames must have matching, nonempty stems")
    names = sorted(images)
    return [images[n] for n in names], [masks[n] for n in names]
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ToTensor(object):

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']

        return {'image': F.resize(image, self.size), 'label': F.resize(label, self.size, interpolation=InterpolationMode.NEAREST)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}
    

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode, with_shape_fields=False,
                 field_epsilon=0.01, field_points=30):
        self.images, self.gts = paired_files(image_root, gt_root)
        if with_shape_fields and (size < 32 or size % 32):
            raise ValueError("CCS training size must be a positive multiple of 32")
        self.with_shape_fields = with_shape_fields
        self.field_epsilon, self.field_points = field_epsilon, field_points
        self.field_sizes = [(size // d, size // d) for d in (16, 8, 4)]
        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        data = {'image': image, 'label': label}
        data = self.transform(data)
        data['label'] = (data['label'] >= 0.5).float()
        if self.with_shape_fields:
            # Build AFTER resize and flips to preserve geometry and vector signs.
            fields, valid = shape_targets(data['label'], self.field_sizes,
                                           self.field_epsilon, self.field_points)
            data['shape_fields'], data['shape_valid'] = fields, valid
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')



class TestDataset:
    def __init__(self, image_root, gt_root, size):
        self.images, self.gts = paired_files(image_root, gt_root)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = os.path.basename(self.images[self.index])

        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
