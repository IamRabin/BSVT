from argparse import ArgumentParser, Namespace
from enum import Enum
from lightning.pytorch.accelerators import find_usable_cuda_devices
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pytorch_lightning.strategies import DeepSpeedStrategy
#import torch

from dataset import TRAIN_DIR, VAL_DIR, ImageFolder


BATCH_SIZE = 8


def main(args):

    #torch.backends.cuda.reserved_memory = 0
    #torch.backends.cuda.max_split_size_mb = 128
    
    strategy=DeepSpeedStrategy()

    transforms = T.Compose([
         #T.ToTensor()
    ])
    train_dataset = ImageFolder(TRAIN_DIR, transform=transforms)
    val_dataset = ImageFolder(VAL_DIR, transform=transforms)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=False,
                              # persistent_workers=True,
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=False,
                            # persistent_workers=True,
                            )

    vargs = vars(args)
    model_name: Model = vargs.get('model')
    print(model_name)

    if model_name == Model.cct500:
         img_size = (19, 500)
         model = CCT(
                 #conv_kernel = 3, conv_stride = 3, conv_pad = 0,
                  #   pool_kernel = 3, pool_stride = 1, pool_pad = 0,
                     heads = 4, emb_dim = 64, feat_dim= 2*64,
                     dropout = 0.1, attention_dropout = 0.1, layers = 4,
                     image_size = img_size, num_class = 2)

         parameters = filter(lambda p: p.requires_grad, model.parameters())
         parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
         print('Trainable Parameters: %.3fM' % parameters)

    


    if model_name == Model.cct118:
         img_size = (118,500)
         model = CCT(
           conv_kernel = 3, conv_stride = 2, conv_pad = 3,
           pool_kernel = 3, pool_stride = 2, pool_pad = 1,
           heads = 4, emb_dim = 384, feat_dim= 2*384,
           dropout = 0.1, attention_dropout = 0.1, layers = 7,
           channels = 19, image_size = img_size, num_class = 2)

         parameters = filter(lambda p: p.requires_grad, model.parameters())
         parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
         print('Trainable Parameters: %.3fM' % parameters)

    #trainer: pl.Trainer = pl.Trainer.from_argparse_args(args,max_epochs=250)
    trainer= pl.Trainer(max_epochs=500, accelerator="gpu", devices=4,
            strategy=strategy,precision="16"
            )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )




class Model(Enum):

    cct500 = 'CCT500'
    cct118 = 'CCT118'

    def __str__(self):
        return self.value


if __name__ == '__main__':
    
    parser = ArgumentParser()
    #parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--model", type=Model, choices=list(Model), required=True)
    args=parser.parse_args()
    main(args)
