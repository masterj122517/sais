import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer


class DisProtDataset(Dataset):
    def __init__(self, dict_data, tokenizer, max_len=512):
        self.sequences = dict_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]["sequence"]
        label = [int(c) for c in self.sequences[idx]["label"]]
        encoded = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        label = torch.tensor(label[: self.max_len], dtype=torch.long)
        if len(label) < self.max_len:
            pad_len = self.max_len - len(label)
            label = F.pad(
                label, (0, pad_len), value=-100
            )  # use -100 for ignored index in loss
        return {key: val.squeeze(0) for key, val in encoded.items()}, label


class DisProtModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "Rostlab/prot_bert", output_hidden_states=False
        )
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, model_config.o_dim)

    def forward(self, inputs):
        output = self.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids", None),
        )
        x = self.dropout(output.last_hidden_state)
        return self.classifier(x)


def make_dataset(data_config, tokenizer, train_rate=0.7, valid_rate=0.2):
    with open(data_config.data_path, "rb") as f:
        data = pickle.load(f)

    total_number = len(data)
    train_sep = int(total_number * train_rate)
    valid_sep = int(total_number * (train_rate + valid_rate))

    train_data = data[:train_sep]
    valid_data = data[train_sep:valid_sep]
    test_data = data[valid_sep:]

    return (
        DisProtDataset(train_data, tokenizer),
        DisProtDataset(valid_data, tokenizer),
        DisProtDataset(test_data, tokenizer),
    )


def metric_fn(pred, gt):
    pred = pred.detach().cpu()
    gt = gt.detach().cpu()
    pred_labels = torch.argmax(pred, dim=-1).view(-1)
    gt_labels = gt.view(-1)
    mask = gt_labels != -100
    score = f1_score(y_true=gt_labels[mask], y_pred=pred_labels[mask], average="micro")
    return score


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser("IDRs prediction with ProtBERT")
    parser.add_argument("--config_path", default="./config.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)

    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

    train_dataset, valid_dataset, test_dataset = make_dataset(config.data, tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, **config.train.dataloader)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)

    model = DisProtModel(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.optimizer.lr,
        weight_decay=config.train.optimizer.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # 初始化评估
    model.eval()
    metric = 0.0
    with torch.no_grad():
        for batch, label in valid_dataloader:
            for k in batch:
                batch[k] = batch[k].to(device)
            label = label.to(device)
            pred = model(batch)
            metric += metric_fn(pred, label)
    print("init f1_score:", metric / len(valid_dataloader))

    # 训练
    for epoch in range(config.train.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, initial=0, desc=f"epoch:{epoch:03d}")
        total_loss = 0.0
        for batch, label in progress_bar:
            for k in batch:
                batch[k] = batch[k].to(device)
            label = label.to(device)

            pred = model(batch)
            loss = loss_fn(pred.permute(0, 2, 1), label)
            progress_bar.set_postfix(loss=loss.item())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(train_dataloader)

        # 验证
        model.eval()
        metric = 0.0
        with torch.no_grad():
            for batch, label in valid_dataloader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                label = label.to(device)
                pred = model(batch)
                metric += metric_fn(pred, label)
        print(
            f"avg_training_loss: {avg_loss}, f1_score: {metric / len(valid_dataloader)}"
        )
