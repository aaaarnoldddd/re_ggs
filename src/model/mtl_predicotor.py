import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import MeanMetric, SpearmanCorrCoef
from pytorch_lightning import LightningModule, Trainer
import hydra
import logging


# LengthMaxPool1D 层：对序列维度做最大池化，可选线性变换与激活
class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim, linear=False, activation='relu'):
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)
        if activation == 'swish':
            self.act_fn = lambda x: x * torch.sigmoid(100.0 * x)
        elif activation == 'softplus':
            self.act_fn = nn.Softplus()
        elif activation == 'sigmoid':
            self.act_fn = nn.Sigmoid()
        elif activation == 'leakyrelu':
            self.act_fn = nn.LeakyReLU()
        elif activation == 'relu':
            self.act_fn = lambda x: F.relu(x)
        else:
            raise NotImplementedError(f"Activation {activation} not supported.")
    
    def forward(self, x):
        if self.linear:
            x = self.act_fn(self.layer(x))
        x = torch.max(x, dim=1)[0]
        return x

# MLP_Block：通用的多层感知机模块，用于构造专家网络、门控网络以及任务塔
class MLP_Block(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim=None, hidden_activations="relu", output_activation=None,
                 dropout_rates=0.0, batch_norm=False):
        super(MLP_Block, self).__init__()
        layers = []
        current_dim = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(current_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            if hidden_activations.lower() == "relu":
                layers.append(nn.ReLU())
            elif hidden_activations.lower() == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                raise NotImplementedError(f"Activation {hidden_activations} not supported.")
            if dropout_rates > 0.0:
                layers.append(nn.Dropout(dropout_rates))
            current_dim = h
        if output_dim is not None:
            layers.append(nn.Linear(current_dim, output_dim))
            if output_activation is not None:
                if output_activation.lower() == "relu":
                    layers.append(nn.ReLU())
                elif output_activation.lower() == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif output_activation.lower() == "identity":
                    layers.append(nn.Identity())
                else:
                    raise NotImplementedError(f"Output activation {output_activation} not supported.")
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

# get_activation 函数用于返回激活层
def get_activation(act_str):
    if act_str.lower() == "softmax":
        return nn.Softmax(dim=1)
    elif act_str.lower() == "relu":
        return nn.ReLU()
    elif act_str.lower() == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError(f"Activation {act_str} not supported in get_activation.")

# MMoE_Layer：多任务专家层
class MMoE_Layer(nn.Module):
    def __init__(self, num_experts, num_tasks, input_dim, expert_hidden_units, gate_hidden_units,
                 hidden_activations, net_dropout, batch_norm):
        super(MMoE_Layer, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([
            MLP_Block(
                input_dim=input_dim,
                hidden_units=expert_hidden_units,
                output_dim=expert_hidden_units[-1],
                hidden_activations=hidden_activations,
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            )
            for _ in range(num_experts)
        ])
        self.gate = nn.ModuleList([
            MLP_Block(
                input_dim=input_dim,
                hidden_units=gate_hidden_units,
                output_dim=num_experts,
                hidden_activations=hidden_activations,
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            )
            for _ in range(num_tasks)
        ])
        self.gate_activation = get_activation('softmax')
    
    def forward(self, x):
        # x: (batch_size, input_dim)
        experts_output = torch.stack([expert(x) for expert in self.experts], dim=1)  # (batch_size, num_experts, expert_dim)
        mmoe_output = []
        for i in range(self.num_tasks):
            gate_output = self.gate[i](x)  # (batch_size, num_experts)
            gate_output = self.gate_activation(gate_output)
            weighted_experts = gate_output.unsqueeze(-1) * experts_output  # (batch_size, num_experts, expert_dim)
            task_output = weighted_experts.sum(dim=1)  # (batch_size, expert_dim)
            mmoe_output.append(task_output)
        return mmoe_output

# MultiTargetCNNMMoE 模型：结合 CNN 和 MMoE 架构实现多目标回归预测
class MultiTargetCNNMMoE(nn.Module):
    def __init__(self,
                 *,
                 n_tokens: int = 20,           # 词汇表大小为 20（取值范围 0~19）
                 kernel_size: int = 5,
                 input_size: int = 32,
                 dropout: float = 0.1,
                 make_one_hot: bool = True,
                 activation: str = 'relu',
                 linear: bool = True,
                 num_tasks: int = 3,           # 多任务数量，例如预测 3 个适应度指标
                 num_experts: int = 3,
                 expert_hidden_units: list = [128, 64],
                 gate_hidden_units: list = [64],
                 tower_hidden_units: list = [64],
                 hidden_activations: str = 'relu',
                 net_dropout: float = 0.1,
                 batch_norm: bool = False,
                 **kwargs):
        super(MultiTargetCNNMMoE, self).__init__()
        self.n_tokens = n_tokens
        self.make_one_hot = make_one_hot
        # Encoder: 对 one-hot 编码的输入进行卷积操作
        self.encoder = nn.Conv1d(in_channels=n_tokens, out_channels=input_size, kernel_size=kernel_size)
        # CNN 底部池化层：LengthMaxPool1D 将卷积输出在序列维度上进行 max pooling，同时可选线性变换
        self.cnn_pool = LengthMaxPool1D(in_dim=input_size, out_dim=input_size * 2, linear=linear, activation=activation)
        self.dropout = nn.Dropout(dropout)
        out_dim = input_size * 2  # 固定长度特征向量的维度
        # MMoE 层：输入为 CNN 池化层输出的特征向量
        self.mmoe_layer = MMoE_Layer(num_experts=num_experts,
                                     num_tasks=num_tasks,
                                     input_dim=out_dim,
                                     expert_hidden_units=expert_hidden_units,
                                     gate_hidden_units=gate_hidden_units,
                                     hidden_activations=hidden_activations,
                                     net_dropout=net_dropout,
                                     batch_norm=batch_norm)
        # 任务塔：每个任务有一个独立的塔，将 MMoE 输出映射为一个标量（回归输出）
        self.tower = nn.ModuleList([
            MLP_Block(
                input_dim=expert_hidden_units[-1],
                hidden_units=tower_hidden_units,
                output_dim=1,
                hidden_activations=hidden_activations,
                output_activation="identity",
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            )
            for _ in range(num_tasks)
        ])
        # 对于回归任务，输出激活直接 Identity
        self.output_activation = nn.ModuleList([nn.Identity() for _ in range(num_tasks)])
    
    def forward(self, x):
        """
        输入:
          x: (batch_size, sequence_length) 每个元素为一个整数 token（范围 0~19）
        输出:
          如果多任务，则返回字典：{"task1_pred": tensor, "task2_pred": tensor, ...}
          如果单任务，则直接返回张量。
        """
        if self.make_one_hot:
            # 将整数序列转换为 one-hot 表示，形状: (batch_size, sequence_length, n_tokens)
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        # 转置为 (batch_size, n_tokens, sequence_length) 适配 Conv1d
        x = x.permute(0, 2, 1).float()
        x = self.encoder(x)             # (batch_size, input_size, new_seq_len)
        x = x.permute(0, 2, 1)            # (batch_size, new_seq_len, input_size)
        x = self.dropout(x)
        features = self.cnn_pool(x)       # (batch_size, input_size*2)
        expert_outputs = self.mmoe_layer(features)  # List of length num_tasks, each: (batch_size, expert_hidden_units[-1])
        task_outputs = [self.tower[i](expert_outputs[i]) for i in range(len(self.tower))]
        task_outputs = [self.output_activation[i](out) for i, out in enumerate(task_outputs)]
        if len(task_outputs) == 1:
            return task_outputs[0]
        else:
            return {f"task{i+1}_pred": task_outputs[i] for i in range(len(task_outputs))}

# ---------------------------
# LightningModule 包装：ProteinFitnessLightningModule
# ---------------------------
from pytorch_lightning import LightningModule

class MTL_Module(LightningModule):
    def __init__(self, mcfg, ocfg):
    # def __init__(self):
        """
        model_cfg 为一个字典，包含模型相关的配置参数，例如：
        {
            "n_tokens": 20,
            "kernel_size": 3,
            "input_size": 32,
            "dropout": 0.1,
            "make_one_hot": True,
            "activation": "relu",
            "linear": True,
            "num_tasks": 3,
            "num_experts": 3,
            "expert_hidden_units": [128, 64],
            "gate_hidden_units": [64],
            "tower_hidden_units": [64],
            "hidden_activations": "relu",
            "net_dropout": 0.1,
            "batch_norm": False,
            "learning_rate": 1e-3
        }
        """
        super().__init__()
        # self.save_hyperparameters(model_cfg)
        self.mcfg = mcfg
        self.ocfg = ocfg
        self.model = MultiTargetCNNMMoE(**self.mcfg)
        # 适应度预测回归任务使用 MSELoss
        self.criterion = nn.MSELoss()
        # 记录训练过程中的指标
        self.train_loss_metric = MeanMetric()
        self.train_sr_metric_1 = SpearmanCorrCoef()
        self.train_sr_metric_2 = SpearmanCorrCoef()
        self.train_sr_metric_3 = SpearmanCorrCoef()
        # 验证集指标
        self.val_loss_metric = MeanMetric()
        self.val_sr_metric_1 = SpearmanCorrCoef()
        self.val_sr_metric_2 = SpearmanCorrCoef()
        self.val_sr_metric_3 = SpearmanCorrCoef()
        # **测试集指标**
        self.test_loss_metric = MeanMetric()
        self.test_sr_metric_1 = SpearmanCorrCoef()
        self.test_sr_metric_2 = SpearmanCorrCoef()
        self.test_sr_metric_3 = SpearmanCorrCoef()
        self._log = logging.getLogger(__name__)
        self.best_train_loss = float('inf')
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        features, targets = batch  # 假设 batch 为 (features, targets)
        pred = self.forward(features)
        targets = torch.stack([torch.tensor(x) for x in targets])
        targets = targets.transpose(0, 1)
        pred_mse = torch.concat([pred[f"task{i+1}_pred"] for i in range(self.mcfg.num_tasks)], dim=1)

        w1, w2, w3 = 0.6, 0.2, 0.2

        loss1 = self.criterion(pred_mse[:, 0], targets[:, 0])
        loss2 = self.criterion(pred_mse[:, 1], targets[:, 1])
        loss3 = self.criterion(pred_mse[:, 2], targets[:, 2])

        # 这里直接线性组合
        loss = w1 * loss1 + w2 * loss2 + w3 * loss3
        self.train_loss_metric(loss)
        self.log("train_loss", self.train_loss_metric, on_step=False, on_epoch=True, prog_bar=True)

        pred_sr_1 = pred["task1_pred"].squeeze()
        pred_sr_2 = pred["task2_pred"].squeeze()
        pred_sr_3 = pred["task3_pred"].squeeze()

        targets_sr_1 = torch.stack([x[0] for x in targets])
        targets_sr_2 = torch.stack([x[1] for x in targets])
        targets_sr_3 = torch.stack([x[2] for x in targets])

        sr_val_1 = self.train_sr_metric_1(pred_sr_1, targets_sr_1)
        sr_val_2 = self.train_sr_metric_2(pred_sr_2, targets_sr_2)
        sr_val_3 = self.train_sr_metric_3(pred_sr_3, targets_sr_3)

        avg_sr = (sr_val_1 + sr_val_2 + sr_val_3) / 3
        self.log("train_sr", avg_sr, on_step=False, on_epoch=True, prog_bar=True)

        if loss.item() < self.best_train_loss:
            self.best_train_loss = loss.item()
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        验证集的计算逻辑，过程类似训练，但不需要反向传播。
        Lightning 会自动在每个 epoch 的验证阶段调用此方法。
        """
        features, targets = batch
        pred_dict = self.forward(features)
        
        preds = torch.cat([pred_dict[f"task{i+1}_pred"] for i in range(self.mcfg.num_tasks)], dim=1)
        targets = torch.stack([torch.tensor(t) for t in targets]).transpose(0, 1)

        w1, w2, w3 = 0.6, 0.2, 0.2
        loss1 = self.criterion(preds[:, 0], targets[:, 0])
        loss2 = self.criterion(preds[:, 1], targets[:, 1])
        loss3 = self.criterion(preds[:, 2], targets[:, 2])
        loss = w1 * loss1 + w2 * loss2 + w3 * loss3
        
        # 更新 val loss
        self.val_loss_metric(loss)

        # Spearman
        sr_val_1 = self.val_sr_metric_1(pred_dict["task1_pred"].squeeze(), targets[:, 0])
        sr_val_2 = self.val_sr_metric_2(pred_dict["task2_pred"].squeeze(), targets[:, 1])
        sr_val_3 = self.val_sr_metric_3(pred_dict["task3_pred"].squeeze(), targets[:, 2])
        avg_sr = (sr_val_1 + sr_val_2 + sr_val_3) / 3
        
        # 这里用 self.log 记录验证指标, on_epoch=True 表示在 epoch_end 汇总
        self.log("val_loss", self.val_loss_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_sr", avg_sr, on_step=False, on_epoch=True, prog_bar=True)

        return loss 

    def test_step(self, batch, batch_idx):
        """
        处理测试集，每个 batch 计算 loss 和 spearman 相关系数
        """
        features, targets = batch
        pred_dict = self.forward(features)

        preds = torch.cat([pred_dict[f"task{i+1}_pred"] for i in range(self.mcfg.num_tasks)], dim=1)
        targets = torch.stack([torch.tensor(t) for t in targets]).transpose(0, 1)

        w1, w2, w3 = 0.6, 0.2, 0.2
        loss1 = self.criterion(preds[:, 0], targets[:, 0])
        loss2 = self.criterion(preds[:, 1], targets[:, 1])
        loss3 = self.criterion(preds[:, 2], targets[:, 2])
        loss = w1 * loss1 + w2 * loss2 + w3 * loss3
        
        # 记录测试损失
        self.test_loss_metric(loss)

        # Spearman 计算
        sr_val_1 = self.test_sr_metric_1(pred_dict["task1_pred"].squeeze(), targets[:, 0])
        sr_val_2 = self.test_sr_metric_2(pred_dict["task2_pred"].squeeze(), targets[:, 1])
        sr_val_3 = self.test_sr_metric_3(pred_dict["task3_pred"].squeeze(), targets[:, 2])
        avg_sr = (sr_val_1 + sr_val_2 + sr_val_3) / 3

        # Logging: Lightning 会自动聚合这些日志
        self.log("test_loss", self.test_loss_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_sr", avg_sr, on_step=False, on_epoch=True, prog_bar=True)

        return loss  # 也可以返回字典，Lightning 会自动汇总

    # def on_test_epoch_end(self):
        """
        在测试集所有 batch 执行完之后自动调用。
        这里拿 metric 计算最终值, 做打印或 self.log 皆可。
        """
        avg_test_loss = self.test_loss_metric.compute()
        avg_sr_1 = self.test_sr_metric_1.compute()
        avg_sr_2 = self.test_sr_metric_2.compute()
        avg_sr_3 = self.test_sr_metric_3.compute()

        avg_sr = (avg_sr_1 + avg_sr_2 + avg_sr_3) / 3

        # 可选：log 到 Lightning (TensorBoard, CSV, etc.)
        # on_test_epoch_end 是测试阶段 => 'test' scope => 会用 "test_*" 命名空间
        self.log("final_test_loss", avg_test_loss, prog_bar=True)
        self.log("final_test_sr", avg_sr, prog_bar=True)

        # 也可以直接 print 出来
        # print(f"[TEST] Average Loss: {avg_test_loss:.4f}, Spearman: {avg_sr:.4f}")

        # 清空 metric state (如果你下一次 test 还要用同一个模块)
        self.test_loss_metric.reset()
        self.test_sr_metric_1.reset()
        self.test_sr_metric_2.reset()
        self.test_sr_metric_3.reset()


    def on_train_end(self):
        """ 训练结束时打印最小 train_loss """
        self._log.info(f"Best train loss during training: {self.best_train_loss:.6f}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.ocfg)
        return optimizer

# ---------------------------
# Main 函数：测试 LightningModule
# ---------------------------
@hydra.main(version_base=None , config_path="../../config/" , config_name="train.yaml")
def main(cfg):
    # 定义模型配置
    # model_cfg = {
    #     "n_tokens": 20,
    #     "kernel_size": 5,
    #     "input_size": 256,
    #     "dropout": 0.1,
    #     "make_one_hot": True,
    #     "activation": "relu",
    #     "linear": True,
    #     "num_tasks": 3,  # 多任务：预测 3 个适应度指标
    #     "num_experts": 3,
    #     "expert_hidden_units": [128, 64],
    #     "gate_hidden_units": [64],
    #     "tower_hidden_units": [64],
    #     "hidden_activations": "relu",
    #     "net_dropout": 0.1,
    #     "batch_norm": False,
    #     "learning_rate": 1e-3
    # }
    
    # 实例化 LightningModule
    model = MTL_Module(cfg.model.mtl, cfg.model.optimizer)
    
    # 构造示例数据：假设 batch_size = 2, sequence_length = 10
    # 输入为整数序列，取值范围 0~19
    batch_size = 5
    seq_len = 237
    x = torch.randint(0, cfg.model.mtl.n_tokens, (batch_size, seq_len))
    # 假设目标为 3 个适应度值（回归）
    targets = torch.randn(batch_size, 3)  # 注意：若多任务则目标 shape 可能为 (batch_size, num_tasks)
    # 这里为了示例，将多任务回归目标合并为单个张量；实际多任务时可修改 batch 数据组织方式
    batch = (x, targets)
    
    # 测试前向传播和 training_step
    loss = model.training_step(batch, 0)
    print("Training step loss:", loss.item())

    
if __name__ == "__main__":
    main()
