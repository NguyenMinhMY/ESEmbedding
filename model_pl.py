import os
import torch
from dataset import ESDataset
from torch.optim import AdamW
import lightning.pytorch as pl
from optim import NoamScheduler
from loss import ContrastiveLoss
from omegaconf import DictConfig
from model_core import ESEmbedding



class PLESEMbedding(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(PLESEMbedding, self).__init__()
        
        self.cfg = cfg
        self.model = ESEmbedding(cfg)
            
        self.optimizer = AdamW(
            params=self.parameters(),
            lr=cfg.optimizer.lr,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
            eps=1e-9,
        )
        self.scheduler = NoamScheduler(
            self.optimizer,
            factor=cfg.optimizer.factor,
            model_size=cfg.optimizer.model_size,
            warmup_steps=cfg.optimizer.warmup_steps,
        )
        
        self.criterion = ContrastiveLoss(cfg['contrastive_margin'])
        
        self.train_data = ESDataset(cfg['train_dataset'])
        self.valid_data = ESDataset(cfg['val_dataset'])
    
    def training_step(self, batch, batch_idx):
        signal_list = batch
        signal_out = self.model(signal_list)
        batch_size = self.cfg.train_dataset.batch_size
        
        loss = self.criterion(signal_out, batch_size=batch_size)
        
        log_dict = {
            "train_loss": {"value": loss, "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True},
            "lr": {
                "value": self.optimizer.param_groups[0]['lr'], 
                "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True
            }
        }
        self._logging(log_dict)

        return loss
    
    def validation_step(self, batch, batch_idx):
        signal_list = batch
        signal_out = self.model(signal_list)
        batch_size = self.cfg.val_dataset.batch_size

        loss = self.criterion(signal_out, batch_size = batch_size)
        
        log_dict = {
            "valid_loss": 
                {"value": loss, 
                 "on_step": True, 
                 "on_epoch": True, 
                 "prog_bar": True, 
                 "logger": True
                },
        }
        self._logging(log_dict)

        return loss
    
    def test_step(self, batch, batch_idx):
        anchors, _, _ = batch
        anchors_out = self.model(anchors)
        os.mkdir('/dump')
        
        for ith, anchor in enumerate(anchors_out):
            anchor.save(f'/dump/emb_{ith}.pt')
    
    def _logging(self, logs: dict):
        for key in logs:
            self.log(
                key,
                logs[key]['value'],
                logs[key]['on_step'],
                logs[key]['on_epoch'],
                logs[key]['prog_bar'],
                logs[key]['logger'],
            )
        
    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }