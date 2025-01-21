import os
import pyrootutils
import logging
import numpy as np
import hydra
from omegaconf import DictConfig,OmegaConf

import torch
from torch import nn
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.utils.data.dataloader

# logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pyrootutils.setup_root(
    search_from = __file__,
    indicator = ["environment.yaml"],
    pythonpath = True
)

from test_src.utils import hello
from test_src.utils import get_logger

# 普通torch建立方式
class linear_model(nn.Module):
    def __init__(self,input_dims,output_dims):
        super().__init__()
        self.linear=nn.Linear(input_dims,output_dims)

    def forward(self,x):
        return self.linear(x)
    

def train_linear(cfg : DictConfig):
    input_dims=cfg.model.input_dims
    output_dims=cfg.model.output_dims
    data_size=cfg.data.size
    data_std=cfg.data.std
    t_w=cfg.parameters.w
    t_b=cfg.parameters.b

    x = torch.randn(data_size , input_dims)
    w = torch.tensor(t_w)
    w = w.reshape(-1,1)
    y = torch.mm(x , w)
    y += data_std*torch.randn(data_size,1)
    y += t_b

    # print(x.shape,y.shape)
    # print(f'x={x[1]}, y={y[1]}')
    dataset = torch.utils.data.TensorDataset(x,y)
    batch_size = cfg.training.batch_size
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # for idx,(tx,ty) in enumerate(dataloader):
    #     print(f"batch {idx}, {tx.shape}, {ty.shape}")

    net=linear_model(input_dims,output_dims)
    loss=nn.MSELoss()
    lr=cfg.training.lr
    opt=optim.SGD(net.parameters(),lr)

    epochs=cfg.training.epochs
    for epoch in range(epochs):
        tot_l=0.0
        for x,y in dataloader:
            y_hat=net(x)
            l=loss(y_hat,y)
            opt.zero_grad()
            l.backward()
            opt.step()
            tot_l += l.item() * x.shape[0]
        tot_l /= data_size
        if (epoch+1) % 10 == 0:
            print(f"epoch {epoch+1}, loss={tot_l}")

    l_w = net.linear.weight.data[0]
    l_b = net.linear.bias.data[0]
    print(f"learned w={l_w}, learned b={l_b}")
    print(f"true_w={t_w}, true_b={t_b}")

# <--------------------------------------------------------------------------------->
# pytorch-lightning建立方式
class linear_pl(pl.LightningModule):
    def __init__(self,input_dims, output_dims, lr):
        super().__init__()
        self.lr=lr
        self.linear=nn.Linear(input_dims, output_dims)
        self.loss=nn.MSELoss()

    def forward(self,x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x,y=batch
        y_hat=self(x)
        l=self.loss(y_hat, y)

        self.log("train_loss", l, on_step=False, on_epoch=True, prog_bar=True)

        return l
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class data_pl(pl.LightningDataModule):
    def __init__(self, data_size, input_dims, output_dims, t_w, t_b, data_std, batch_size):
        super().__init__()
        self.data_size=data_size
        self.input_dims=input_dims
        self.output_dims=output_dims
        self.t_w=torch.tensor(t_w).reshape(-1,1)
        self.t_b=t_b
        self.data_std=data_std
        self.batch_size=batch_size

    def setup(self,stage=None):
        self.x = torch.randn(self.data_size, self.input_dims)
        self.y = torch.mm(self.x, self.t_w)
        self.y += self.t_b
        self.y += self.data_std * torch.randn(self.data_size, self.output_dims)

        self.dataset=torch.utils.data.TensorDataset(self.x, self.y)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True
        )


def train_linear_pl(cfg):
    input_dims=cfg.model.input_dims
    output_dims=cfg.model.output_dims
    data_size=cfg.data.size
    data_std=cfg.data.std
    t_w=cfg.parameters.w
    t_b=cfg.parameters.b

    batch_size = cfg.training.batch_size
    lr = cfg.training.lr
    epochs=cfg.training.epochs

    my_callbacks=hydra.utils.instantiate(cfg.callbacks)



    data = data_pl(data_size, input_dims, output_dims, t_w, t_b, data_std, batch_size)
    net = linear_pl(input_dims, output_dims, lr)

    trainer = Trainer(max_epochs=epochs,callbacks=my_callbacks,accelerator="gpu")
    trainer.fit(net, data)

    l_w = net.linear.weight.data[0]
    l_b = net.linear.bias.data[0]
    print(f"learned w={l_w}, learned b={l_b}")
    print(f"true_w={t_w}, true_b={t_b}")

    pre_net = linear_pl.load_from_checkpoint(
        checkpoint_path = "ckpt/linear/linear_pl_model_epoch=60_train_loss=0.00.ckpt", 
        **cfg.model)

    test_data = torch.rand(10,2).to(device) * 10

    pre_net.to(device)
    pre_net.eval()

    with torch.no_grad():
        predictions = pre_net(test_data)

    for ((x1, x2), y) in zip(test_data, predictions):
        print(x1.item(), x2.item(), y.item())


    ckpt_data = torch.load("ckpt/linear/linear_pl_model_epoch=60_train_loss=0.00.ckpt")

    # 查看保存的内容
    print(ckpt_data.keys())  # 查看包含哪些内容
    print(ckpt_data['state_dict']) 





@hydra.main(version_base=None , config_path="." , config_name="config.yaml")
def main(cfg : DictConfig):
    logger=get_logger(__name__)
    print(hello())
    print("CONFIG")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    # train_linear(cfg)
    # train_linear_pl(cfg)
    # print(len("SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDTTYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPVGYVLERTIFFKDDGNYKTRAVVKFEGDTLVNRIELKGIDFKEDGNVLGHKLEYNYNSHNVYIMAGKQRNGIKVNFKIRHNIEDGSVQLADHNQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"))
    


    

if __name__ == "__main__":
    main()