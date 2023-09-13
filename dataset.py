import os
import os.path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset

TRAIN_DIR = "/home/rabink1/D1/vtf_images/tuh/train"
VAL_DIR = "/home/rabink1/D1/vtf_images/tuh/eval"
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.npy')


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class DatasetFolder(VisionDataset):
    def __init__(self, root, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError(
                "Found 0 files in subfolders of: " + self.root + "\nSupported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target=np.repeat(target,sample.size(0),axis =0)
            target = self.target_transform(target)
            

        return sample, target

    def __len__(self):
        return len(self.samples)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)

        return img.convert('RGB')


def npy_loader(path):
    img = np.load(path).astype("float32")
    img= np.expand_dims(img, axis=0)
    #z-score
    #img_z=(img - np.mean(img) )/ np.std(img)
    #normalize between 0 and 1
    #img_scaled=(img- np.min(img)) / (np.max(img) - np.min(img))
    return img


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=npy_loader, is_valid_file=None):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

            




if __name__ == '__main__':
    train_transforms = T.Compose([
        #T.ToTensor()
    ])
    val_transforms = T.Compose([
        #T.ToTensor()

    ])

    train_dataset = ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = ImageFolder(VAL_DIR, transform=val_transforms)

    train_loader = DataLoader(train_dataset,
                              batch_size=2,
                              shuffle=True,
                              # num_workers=1,
                              # pin_memory=True if device else False
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=2,
                            shuffle=True,
                            # num_workers=1,
                            # pin_memory=True if device else False
                            )

    print(train_loader.dataset.classes)
    print(train_loader.dataset.class_to_idx)
    

    x, y = next(iter(train_loader))
    print("######################### X ####################")
    print(x.shape)
    print("######################### Y ####################")
    print(y.shape)
    print("######################## ####################")
    print(y)
