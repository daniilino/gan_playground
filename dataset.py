from torchvision import datasets
import torch
import torch.multiprocessing as mp

from torchvision import transforms as T
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data_utils import calculate_mean_std, split_train_val_test

class TwoImageDataset(datasets.vision.VisionDataset):

    """

    """
    
    def __init__(self, path_img1: str, path_img2: str, transform=None, num_samples=2) -> None:

        super().__init__(root="")
        if transform is None: self.transform = lambda x: x
        else: self.transform = transform

        self.loader = datasets.folder.pil_loader
        self.samples = [path_img1, path_img2] * (num_samples // 2) 
        self.emb_types = [0, 1] * (num_samples // 2)

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int):
        # Remember: output shape is [3, H, W], there's no batch dim yet!!!!

        dist = 2

        t = self.emb_types[index]
        emb = torch.randn((1,)).abs()
        if t==1:
            emb = dist - emb
        emb = torch.clamp(emb , 0, dist)

        image = self.loader(self.samples[index])
        image = self.transform(image)

        return image, emb

class LitTwoImageDataModule(pl.LightningDataModule):
    def __init__(self, path_img1, path_img2, batch_size=128, dataset_class=TwoImageDataset, user_transforms=None):
        super().__init__()
        self.batch_size = batch_size
        self.transforms = None
        self.user_transforms = user_transforms

        self.raw_dataset = None
        self.dataloaders = None
        self.dataset = None

        self.image_paths = [path_img1, path_img2]

        self.dataset_class = dataset_class

    def setup(self, image_h, image_w, data_mean=None, data_std=None, num_workers=None):

        if num_workers is None:
            available_cores = mp.cpu_count() - 1
        else:
            available_cores = num_workers

        self.transforms = T.Compose([T.ToTensor(), T.Resize((image_h, image_w)), T.Grayscale()])
        self.raw_dataset = self.dataset_class(self.image_paths[0], self.image_paths[1], transform = self.transforms)

        if data_mean is None or data_std is None:
            data_mean, data_std = calculate_mean_std(self.raw_dataset)

        self.transforms.transforms.append(T.Normalize(mean=data_mean,std=data_std))
        if self.user_transforms is not None:
            self.transforms.transforms.extend(self.user_transforms.transforms)

        self.dataset = self.dataset_class(self.image_paths[0], self.image_paths[1], 
                                          transform = self.transforms, 
                                          num_samples=self.batch_size*1024)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, 
                                     num_workers=available_cores, shuffle=True)
        
class LitImageFolderDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir, batch_size=128, dataset_class=ImageFolder, user_transforms=None):
        super().__init__()
        self.batch_size = batch_size
        self.transforms = None
        self.user_transforms = user_transforms

        self.raw_dataset = None
        self.dataloader = None
        self.micro_dataloader = None
        self.dataset = None

        self.dataset_dir = dataset_dir

        self.dataset_class = dataset_class

    def setup(self, image_h, image_w, data_mean=None, data_std=None, num_workers=None):
        if num_workers is None:
            available_cores = mp.cpu_count() - 1
        else:
            available_cores = num_workers

        self.transforms = T.Compose([T.ToTensor(), T.Resize((image_h, image_w)), T.Grayscale()])
        self.raw_dataset = self.dataset_class(self.dataset_dir, transform = self.transforms)

        if data_mean is None or data_std is None:
            data_mean, data_std = calculate_mean_std(self.raw_dataset)

        self.transforms.transforms.append(T.Normalize(mean=data_mean,std=data_std))
        if self.user_transforms is not None:
            self.transforms.transforms.extend(self.user_transforms.transforms)

        self.dataset = self.dataset_class(self.dataset_dir, 
                                          transform = self.transforms)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, 
                                     num_workers=available_cores, shuffle=True)
        
        micro_portion = (self.batch_size / len(self.dataset))*2
        self.micro_dataloader = split_train_val_test(self.dataset, val=micro_portion, test=micro_portion, batch_size=self.batch_size, num_workers=available_cores)["test"]