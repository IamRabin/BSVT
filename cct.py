
import torch
import torch.nn as nn
import pytorch_lightning as pl
from ptflops import get_model_complexity_info
from einops import rearrange
import math

from torchmetrics import ConfusionMatrix
import torchmetrics
from torch.functional import F
from sklearn.metrics import roc_auc_score, roc_curve


import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from thop import profile
from thop import clever_format

#from utils.convtoken_2_plus_1d import ConvTokenizer
from utils.convtokenizer import ConvTokenizer
from utils.seqpool import SeqPool
from utils.transformer_enc import TransformerEncoderBlock


class CCT(pl.LightningModule):
   
    def __init__(
        self,
        #conv_kernel: int = 3, conv_stride: int = 2, conv_pad: int = 3,
        #pool_kernel: int = 3, pool_stride: int = 2, pool_pad: int = 1,
        heads: int = 4, emb_dim: int = 64, feat_dim: int = 2*64, 
        dropout: float = 0.1, attention_dropout: float = 0.1, layers: int = 4, 
        channels: int = 118, image_size: int = (19,500), num_class: int = 2
        ):
        super().__init__()
        self.emb_dim = emb_dim
        self.image_size = image_size
        
        #self.validation_step_outputs = []
        self.cm = ConfusionMatrix(num_classes=num_class,task="binary")
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_auroc = torchmetrics.AUROC(task="binary",average="micro")
        self.tokenizer = ConvTokenizer(channels=118)

        with torch.no_grad():
            x = torch.randn([1,channels, image_size[0], image_size[1]])
            #x = torch.randn([1,channels, image_size[0], image_size[1]])
            out = self.tokenizer(x)
            #_, _, temp_c,ph_c, pw_c  = out.shape
            _, _,ph_c, pw_c  = out.shape

        self.linear_projection = nn.Linear(
                #temp_c,ph_c, pw_c, self.emb_dim
                 ph_c, pw_c, self.emb_dim)

        self.pos_emb = nn.Parameter(
            torch.randn(
               [1, ph_c*pw_c, self.emb_dim]
               # [1, temp_c*ph_c*pw_c, self.emb_dim]
            ).normal_(std=0.02) # from torchvision, which takes this from BERT
        )
        self.dropout = nn.Dropout(dropout)
        encoders = []
        for _ in range(0, layers):
            encoders.append(
                TransformerEncoderBlock(
                    n_h=heads, emb_dim=self.emb_dim, feat_dim=feat_dim,
                    dropout=dropout, attention_dropout=attention_dropout
                )
            )
        self.encoder_stack = nn.Sequential(*encoders)
        self.seq_pool = SeqPool(emb_dim=self.emb_dim)
        self.mlp_head = nn.Linear(self.emb_dim, num_class)


    def forward(self, x: torch.Tensor):     
        #bs,c,t, h, w = x.shape  # (bs,c,t, h, w)
        bs, c, h, w = x.shape  # (bs,c, h, w)

        # Creates overlapping patches using ConvNet
        x = self.tokenizer(x)
        print(f"token shape:{x.shape}")
        x = rearrange(
            x, 'bs e_d ph_h ph_w -> bs ( ph_h ph_w) e_d ',    
            bs=bs, e_d=self.emb_dim
        )

        # Add position embedding
        x = self.pos_emb.expand(bs, -1, -1) + x
        x = self.dropout(x)

        # Pass through Transformer Encoder layers
        x = self.encoder_stack(x)

        # Perform Sequential Pooling <- Novelty of the paper
        x = self.seq_pool(x)

        # MLP head used to get logits
        x = self.mlp_head(x)

        return x

    def configure_optimizers(self):
       # return torch.optim.Adam(self.parameters(), lr=0.0001)
        optimizer=torch.optim.SGD(self.parameters(), lr=0.01,momentum=0.9)
        sch = torch.optim.lr_scheduler.CyclicLR(optimizer,cycle_momentum=True, base_lr=0.000001, max_lr=0.01)
        #learning rate scheduler
        return {
            "optimizer":optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "train_loss",

            }
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self(x)
        _, preds = torch.max(y_hat.data, 1)

        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, on_step=False,sync_dist=True)

        self.train_acc(preds, y)

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss,sync_dist=True)

        _, preds = torch.max(y_hat.data, 1)
        #self.val_auroc.update(pred, y)
        self.cm.update(preds, y)
        self.val_acc(preds, y)
        self.val_auroc.update(preds, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True,sync_dist=True)
        #self.validation_step_outputs.append(preds)
    
        return {'val_loss': loss, 'y_true': y, 'y_pred': preds}

    


    def on_validation_epoch_end(self):
        # TODO num classes
        df_cm = pd.DataFrame(self.cm.compute().cpu().data, index=range(2), columns=range(2))
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)



        self.logger.experiment.add_figure("Validation confusion matrix", fig_, self.current_epoch)
        self.log('valid_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        
        val_auroc = self.val_auroc.compute()
        self.log("val_auroc", val_auroc,on_epoch=True,on_step=False,sync_dist=True)
        

#
#        # calculate ROC curve
#        all_y_true = torch.cat([x['y_true'] for x in self.outputs])
#        all_y_pred = torch.cat([x['y_pred'] for x in self.outputs])
#        fpr, tpr, _ = roc_curve(all_y_true.cpu().numpy(), all_y_pred.cpu().numpy())
#
#        # plot ROC curve
#        fig, ax = plt.subplots()
#        ax.plot(fpr, tpr)
#        ax.set_xlabel('False Positive Rate')
#        ax.set_ylabel('True Positive Rate')
#        ax.set_title('ROC Curve')
#        self.logger.experiment.add_figure('roc_curve', fig, self.current_epoch)
#


  
  
  
  

  
  
  
  












if __name__=="__main__":
   #x = torch.randn(2,1,118,19,500)
   x=torch.rand(2,118,19,500)
   t = CCT(channels=118)
   print(t(x).shape )
   #parameters = filter(lambda p: p.requires_grad, t.parameters())
   #parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
   #print('Trainable Parameters: %.3fM' % parameters)
   #macs,_ = profile(t, inputs=(x,))
   #macs, _ = clever_format([macs], "%.3f")
   #print(f"MACs: {macs}")
   ############

  
   with torch.cuda.device(0):
            macs, params = get_model_complexity_info(t, (118,19,500), as_strings=True, print_per_layer_stat=False)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))



 
