import pickle
import gensim
import gensim.models

import os
import sys
import numpy as np
import pandas as pd
from joblib import load, dump

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report


def predict_test(path):
    model, model_w2v = load("model.pkl")
    test_datas = pickle.load(open(path, "rb"))
    submit_data = []
    for data in test_datas:
        data_x = []
        sequence = list(data["sequence"])
        for idx, _ in enumerate(sequence):
            data_x.append(
                model_w2v.wv[
                    sequence[max(0, idx - 2) : min(len(sequence), idx + 2)]
                ].mean(0)
            )
        data_x = np.array(data_x)
        pred = model.predict(data_x)

        submit_data.append(
            [data["id"], data["sequence"], "".join([str(c) for c in pred])]
        )

    submit_df = pd.DataFrame(submit_data)
    submit_df.columns = ["proteinID", "sequence", "IDRs"]
    submit_df.to_csv("/saisresult/submit.csv", index=None)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train_model()
    else:
        predict_test("/saisdata/WSAA_data_test.pkl")
