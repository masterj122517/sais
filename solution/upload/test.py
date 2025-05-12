import argparse
import pickle
import numpy as np


def predict_test(test_path, model_path="model.pkl", output_path="result.csv"):
    from sklearn.linear_model import LogisticRegression
    from gensim.models import Word2Vec

    # 读取测试数据
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    sequences = [list(d["sequence"]) for d in test_data]

    # 加载训练好的模型和 Word2Vec 模型
    with open(model_path, "rb") as f:
        model, model_w2v = pickle.load(f)

    results = []
    for seq in sequences:
        for idx in range(len(seq)):
            # 提取当前词的上下文（前两个 + 后两个）
            context = seq[max(0, idx - 2) : min(len(seq), idx + 2)]
            vectors = []
            for word in context:
                if word in model_w2v.wv:
                    vectors.append(model_w2v.wv[word])
            if not vectors:
                vec = np.zeros(model_w2v.vector_size)
            else:
                vec = np.mean(vectors, axis=0)
            pred = model.predict([vec])[0]
            results.append(pred)

    # 写入 CSV 文件
    with open(output_path, "w") as f:
        f.write("Id,Predicted\n")
        for idx, label in enumerate(results):
            f.write(f"{idx},{label}\n")

    print(f"预测完成，结果已保存至 {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用训练好的模型进行预测")
    parser.add_argument("--test", required=True, help="测试集路径（pkl 文件）")
    parser.add_argument("--model", default="model.pkl", help="模型文件路径")
    parser.add_argument("--output", default="result.csv", help="输出 CSV 路径")
    args = parser.parse_args()

    predict_test(args.test, args.model, args.output)
