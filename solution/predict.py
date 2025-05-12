import os
import pickle
import torch
import pandas as pd

from main import DisProtModel, DisProtDataset, residue_to_idx, default_idx
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def load_test_data(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    dataset = DisProtDataset(data)
    return dataset


def predict():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = OmegaConf.load("./config.yaml")
    model = DisProtModel(config.model)
    model.load_state_dict(torch.load("./model.pkl", map_location=device))
    model = model.to(device)
    model.eval()

    test_dataset = load_test_data("/saisdata/WSAA_data_test.pkl")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    results = []
    with torch.no_grad():
        for idx, (sequence, _) in enumerate(test_loader):
            sequence = sequence.to(device)
            pred = model(sequence)
            pred_label = torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist()
            if isinstance(pred_label, int):
                pred_label = [pred_label]
            results.append({"id": idx, "pred": "".join(str(i) for i in pred_label)})

    os.makedirs("/saisresult", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("/saisresult/submit.csv", index=False)
    print("预测结果已保存至 /saisresult/submit.csv")


if __name__ == "__main__":
    predict()
