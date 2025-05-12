main.py

1-4:  
导入标准库：argparse（命令行参数解析）、math（数学函数）、pickle（序列化/反序列化）。

5-7:  
导入 PyTorch 相关模块：torch（核心）、torch.nn（神经网络）、torch.nn.functional（函数式接口，未使用）。

9:  
导入 tqdm 进度条库。

10:  
导入 omegaconf 的 OmegaConf，用于读取 yaml 配置。

11:  
导入 sklearn 的 f1_score 评价指标。

12:  
导入 PyTorch 的 Dataset 和 DataLoader。

13:  
导入 TransformerEncoderLayer 和 TransformerEncoder。

15-38:  
定义氨基酸残基类型列表 restypes，以及残基到索引的映射 residue_to_idx，和默认索引 default_idx。

43-61:  
定义 make_dataset 函数：  
- 读取 pickle 格式的数据文件  
- 按比例划分为训练、验证、测试集  
- 返回 DisProtDataset 实例

63-92:  
定义 DisProtDataset 类（继承自 Dataset）：  
- 构造函数读取序列和标签  
- __len__ 返回样本数  
- __getitem__ 返回序列和标签的 tensor 表示

94-112:  
定义 PositionalEncoding 类：  
- 构造函数生成位置编码  
- forward 方法将位置编码加到输入上

114-147:  
定义 DisProtModel 类：  
- 构造函数初始化嵌入层、位置编码、归一化、dropout、transformer 编码器和输出层  
- forward 方法依次通过这些层

149-156:  
定义 metric_fn 函数：  
- 计算预测和真实标签的 micro f1 分数

158-221:  
主程序入口：  
- 判断设备（cuda 或 cpu）  
- 解析命令行参数，读取配置  
- 构建数据集和 DataLoader  
- 初始化模型、优化器、损失函数  
- 先在验证集上评估初始 f1 分数  
- 训练若干 epoch，每个 epoch：  
  - 训练模型，记录损失  
  - 在验证集上评估 f1 分数  
  - 保存模型参数为 model.pkl

predict.py

1-2: 导入os和pickle模块，分别用于文件操作和序列化/反序列化数据。
3: 导入torch库，用于深度学习相关操作。
4: 导入pandas库，用于数据处理和保存结果。
5: 从omegaconf库导入OmegaConf，用于读取yaml配置文件。
6: 从torch.utils.data导入Dataset和DataLoader，用于自定义数据集和批量加载数据。
7: 导入datetime模块（未使用，可删除）。
8: 导入math模块，用于数学计算。
10-34: 定义氨基酸类型列表restypes，以及氨基酸到索引的映射residue_to_idx和默认索引default_idx。
39-56: 定义DisProtDataset类，继承自Dataset，用于处理蛋白质序列数据。
  - 40-48: 初始化方法，读取数据字典，提取序列和蛋白质ID。
  - 49-51: 返回数据集长度。
  - 52-55: 根据索引获取序列的索引表示和蛋白质ID。
58-74: 定义PositionalEncoding类，实现位置编码，用于Transformer模型。
  - 59-70: 初始化方法，生成位置编码矩阵。
  - 71-73: 前向传播，将位置编码加到输入上并做dropout。
76-110: 定义DisProtModel类，继承自nn.Module，实现蛋白质序列的Transformer模型。
  - 77-100: 初始化方法，定义嵌入层、位置编码、归一化、dropout、Transformer编码器和输出层。
  - 102-109: 前向传播，依次通过各层得到输出。
112-118: 定义load_test_data函数，加载测试数据并返回数据集对象。
121-176: 定义predict函数，执行预测并保存结果。
  - 123: 判断是否有GPU可用，设置设备。
  - 126: 加载配置文件。
  - 129-132: 初始化模型并加载参数，设置为评估模式。
  - 135-136: 加载测试数据并创建DataLoader。
  - 138-167: 遍历测试数据，模型预测，处理预测结果，保存到results列表。
  - 169: 创建保存结果的文件夹。
  - 172-175: 将结果保存为CSV文件，并打印提示信息。
178-179: 如果作为主程序运行，则调用predict函数。

