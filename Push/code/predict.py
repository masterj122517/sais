import os
import pickle
import torch
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
import datetime
import math

# 定义氨基酸类型
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


class DisProtDataset(Dataset):
    def __init__(self, dict_data):
        self.sequences = []
        self.protein_ids = []  # 添加protein_ids列表
        for d in dict_data:
            if "sequence" not in d:
                continue
            self.sequences.append(d["sequence"])
            self.protein_ids.append(d.get("id", ""))  # 使用'id'而不是'proteinID'

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        indices = [residue_to_idx.get(c, default_idx) for c in self.sequences[idx]]
        sequence = torch.tensor(indices, dtype=torch.long)
        return sequence, self.protein_ids[idx]  # 返回序列和proteinID


class PositionalEncoding(torch.nn.Module):
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
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class DisProtModel(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.d_model = model_config.d_model
        self.n_head = model_config.n_head
        self.n_layer = model_config.n_layer
        self.vocab_size = len(restypes)

        # 改进的embedding层
        self.embedding = torch.nn.Embedding(self.vocab_size, model_config.d_model)
        self.embedding_dropout = torch.nn.Dropout(p=0.2)  # 增加dropout率
        
        # 添加LayerNorm
        self.embedding_norm = torch.nn.LayerNorm(self.d_model)
        
        self.position_embed = PositionalEncoding(
            self.d_model, dropout=0.1, max_len=40000
        )
        self.input_norm = torch.nn.LayerNorm(self.d_model)
        self.dropout_in = torch.nn.Dropout(p=0.2)  # 增加dropout率

        # 改进的transformer配置
        encoder_layer = torch.nn.TransformerEncoderLayer(
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
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layer
        )
        
        # 改进的输出层
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.d_model * 2),
            torch.nn.LayerNorm(self.d_model * 2),  # 添加LayerNorm
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.2),  # 增加dropout率
            torch.nn.Linear(self.d_model * 2, self.d_model),
            torch.nn.LayerNorm(self.d_model),  # 添加LayerNorm
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.2),  # 增加dropout率
            torch.nn.Linear(self.d_model, model_config.o_dim),
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


def load_test_data(data_path):
    """加载测试数据"""
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    print("测试数据示例：", data[0] if data else "空数据")  # 添加调试信息
    dataset = DisProtDataset(data)
    return dataset


def predict():
    """进行预测并保存结果"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载配置文件
    config = OmegaConf.load("./config.yaml")

    # 初始化模型
    model = DisProtModel(config.model)
    model.load_state_dict(torch.load("./model.pkl", map_location=device))
    model = model.to(device)
    model.eval()  # 设置为评估模式

    # 加载测试数据
    test_dataset = load_test_data("/saisdata/WSAA_data_test.pkl")
    test_loader = DataLoader(test_dataset, batch_size=config.train.dataloader.batch_size, shuffle=False)

    results = []
    with torch.no_grad():  # 禁用梯度计算
        for sequence, protein_id in test_loader:
            sequence = sequence.to(device)  # 将数据转移到设备上
            pred = model(sequence)  # 模型预测

            # 获取预测结果
            pred_label = torch.argmax(pred, dim=-1).cpu().numpy()
            
            # 处理每个样本的预测结果
            for i in range(len(protein_id)):
                # 将预测标签转换为字符串
                idrs_str = "".join(str(int(x)) for x in pred_label[i])
                
                # 获取原始序列
                original_sequence = test_dataset.sequences[len(results)]
                
                # 添加结果
                results.append({
                    "proteinID": protein_id[i],
                    "sequence": original_sequence,
                    "IDRs": idrs_str,
                })

    # 创建保存结果的文件夹
    os.makedirs("/saisresult", exist_ok=True)

    # 将结果保存为 CSV
    df = pd.DataFrame(results)
    print("结果示例：", df.head())  # 添加调试信息
    df.to_csv("/saisresult/submit.csv", index=False)
    print("预测结果已保存至 /saisresult/submit.csv")


if __name__ == "__main__":
    predict()
