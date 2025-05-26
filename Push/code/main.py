import argparse
import math
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoderLayer, TransformerEncoder

restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",
    "U",
]
residue_to_idx = {res: i for i, res in enumerate(restypes)}
default_idx = residue_to_idx["X"]


def make_dataset(data_config, train_rate=0.7, valid_rate=0.2):
    data_path = data_config.data_path
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    total_number = len(data)
    train_sep = int(total_number * train_rate)
    valid_sep = int(total_number * (train_rate + valid_rate))

    train_data_dicts = data[:train_sep]
    valid_data_dicts = data[train_sep:valid_sep]
    test_data_dicts = data[valid_sep:]

    train_dataset = DisProtDataset(train_data_dicts)
    valid_dataset = DisProtDataset(valid_data_dicts)
    test_dataset = DisProtDataset(test_data_dicts)

    return train_dataset, valid_dataset, test_dataset


class DisProtDataset(Dataset):
    def __init__(self, dict_data):
        sequences = []
        labels = []

        for d in dict_data:
            if "sequence" not in d:
                # print(f"Missing 'sequence' key in data: {d}")
                continue  # Skip invalid data
            if "label" not in d:
                # print(f"Missing 'label' key in data: {d}")
                continue  # Skip invalid data

            sequences.append(d["sequence"])
            labels.append(d["label"])

        assert len(sequences) == len(labels)

        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        indices = [residue_to_idx.get(c, default_idx) for c in self.sequences[idx]]
        sequence = torch.tensor(indices, dtype=torch.long)
        label = torch.tensor([int(c) for c in self.labels[idx]], dtype=torch.long)
        return sequence, label


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=40000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class DisProtModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.d_model = model_config.d_model
        self.n_head = model_config.n_head
        self.n_layer = model_config.n_layer
        self.vocab_size = len(restypes)

        # 改进的embedding层
        self.embedding = nn.Embedding(self.vocab_size, model_config.d_model)
        self.embedding_dropout = nn.Dropout(p=0.2)  # 增加dropout率

        # 添加LayerNorm
        self.embedding_norm = nn.LayerNorm(self.d_model)

        self.position_embed = PositionalEncoding(
            self.d_model, dropout=0.1, max_len=40000
        )
        self.input_norm = nn.LayerNorm(self.d_model)
        self.dropout_in = nn.Dropout(p=0.2)  # 增加dropout率

        # 改进的transformer配置
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_head,
            dim_feedforward=self.d_model * 4,
            dropout=0.2,  # 增加dropout率
            activation="gelu",
            batch_first=True,
            norm_first=True,
            # 添加额外的参数
            layer_norm_eps=1e-6,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=self.n_layer)

        # 改进的输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.LayerNorm(self.d_model * 2),  # 添加LayerNorm
            nn.GELU(),
            nn.Dropout(p=0.2),  # 增加dropout率
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),  # 添加LayerNorm
            nn.GELU(),
            nn.Dropout(p=0.2),  # 增加dropout率
            nn.Linear(self.d_model, model_config.o_dim),
        )

    def forward(self, x):
        # 改进的前向传播
        x = self.embedding(x)
        x = self.embedding_norm(x)  # 添加LayerNorm
        x = self.embedding_dropout(x)
        x = self.position_embed(x)
        x = self.input_norm(x)
        x = self.dropout_in(x)

        # 添加残差连接
        residual = x
        x = self.transformer(x)
        x = x + residual

        x = self.output_layer(x)
        return x


def detailed_metric_fn(pred, gt):
    pred = pred.detach().cpu()
    gt = gt.detach().cpu()
    pred_labels = torch.argmax(pred, dim=-1).view(-1)
    gt_labels = gt.view(-1)

    # 计算TP, FP, FN
    tp = ((pred_labels == 1) & (gt_labels == 1)).sum().item()
    fp = ((pred_labels == 1) & (gt_labels == 0)).sum().item()
    fn = ((pred_labels == 0) & (gt_labels == 1)).sum().item()
    tn = ((pred_labels == 0) & (gt_labels == 0)).sum().item()

    # 打印调试信息
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 使用sklearn的f1_score进行验证
    sklearn_f1 = f1_score(gt_labels.numpy(), pred_labels.numpy(), average="binary")

    # 计算F1 score
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # 打印调试信息
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Calculated F1: {f1:.4f}, Sklearn F1: {sklearn_f1:.4f}")

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return {
        "f1": sklearn_f1,  # 使用sklearn的F1 score
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建checkpoints目录
    os.makedirs("checkpoints", exist_ok=True)
    
    # 加载配置
    config = OmegaConf.load("./config.yaml")

    # 创建数据集和数据加载器
    train_dataset, valid_dataset, test_dataset = make_dataset(config.data)
    train_dataloader = DataLoader(dataset=train_dataset, **config.train.dataloader)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)

    # 初始化模型
    model = DisProtModel(config.model)
    model = model.to(device)

    # 使用AdamW优化器，添加权重衰减
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.optimizer.lr,
        weight_decay=0.01,  # 增加权重衰减
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # 使用OneCycleLR学习率调度器，添加warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.train.optimizer.lr,
        epochs=config.train.epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1,  # 10%的warmup
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
        three_phase=True,  # 使用三阶段学习率调度
    )

    # 使用标签平滑的交叉熵损失
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 梯度裁剪
    max_grad_norm = 1.0

    # 增加早停的耐心值
    best_f1 = 0.0
    patience = 20  # 增加早停的耐心值
    patience_counter = 0

    # 记录所有checkpoint的信息
    checkpoint_history = []

    # 初始化评估
    model.eval()
    metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    with torch.no_grad():
        for sequence, label in valid_dataloader:
            sequence = sequence.to(device)
            label = label.to(device)
            pred = model(sequence)
            batch_metrics = detailed_metric_fn(pred, label)
            for k in metrics:
                metrics[k] += batch_metrics[k]

    for k in metrics:
        metrics[k] /= len(valid_dataloader)

    print("初始评估指标:")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    for epoch in range(config.train.epochs):
        progress_bar = tqdm(train_dataloader, initial=0, desc=f"epoch:{epoch:03d}")
        model.train()
        total_loss = 0.0

        # 添加梯度累积
        accumulation_steps = 4  # 每4步更新一次参数
        optimizer.zero_grad()

        for i, (sequence, label) in enumerate(progress_bar):
            sequence = sequence.to(device)
            label = label.to(device)

            pred = model(sequence)
            loss = loss_fn(pred.permute(0, 2, 1), label)
            loss = loss / accumulation_steps  # 缩放损失
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item() * accumulation_steps)
            total_loss += loss.item() * accumulation_steps

        avg_loss = total_loss / len(train_dataloader)

        # 验证评估
        model.eval()
        metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
        with torch.no_grad():
            for sequence, label in valid_dataloader:
                sequence = sequence.to(device)
                label = label.to(device)
                pred = model(sequence)
                batch_metrics = detailed_metric_fn(pred, label)
                for k in metrics:
                    metrics[k] += batch_metrics[k]

        for k in metrics:
            metrics[k] /= len(valid_dataloader)

        print(f"\nEpoch {epoch + 1} 评估结果:")
        print(f"训练损失: {avg_loss:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "f1_score": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "accuracy": metrics["accuracy"],
            "loss": avg_loss,
        }

        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        checkpoint_history.append(
            {"epoch": epoch + 1, "f1_score": metrics["f1"], "path": checkpoint_path}
        )

        # 保存最佳模型
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), "model.pkl")
            print(f"模型已保存为 model.pkl (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # 训练结束后，找出最佳checkpoint
    best_checkpoint = max(checkpoint_history, key=lambda x: x["f1_score"])
    print(f"\n最佳模型信息:")
    print(f"Epoch: {best_checkpoint['epoch']}")
    print(f"F1 Score: {best_checkpoint['f1_score']:.4f}")
    print(f"Checkpoint路径: {best_checkpoint['path']}")

    # 将最佳模型复制为最终模型
    best_model = torch.load(best_checkpoint["path"])
    torch.save(best_model["model_state_dict"], "model.pkl")
    print("最佳模型已保存为 model.pkl")


if __name__ == "__main__":
    train()
