# %%
import pandas as pd
from sklearn import model_selection
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import numpy as np
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, random_split
from sklearn.model_selection import train_test_split
import torch.nn.init as init
import torch.nn.functional as F
import os
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, average_precision_score

# %%
class ColumnAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ColumnAttention, self).__init__()
        # 自注意力的权重
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # 输入 x 维度: (batch_size, 40, 13)
        Q = self.query(x)  # (batch_size, 40, hidden_dim)
        K = self.key(x)    # (batch_size, 40, hidden_dim)
        V = self.value(x)  # (batch_size, 40, hidden_dim)
        
        # 计算注意力分数: Q * K^T
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, 40, 40)
        attention_scores = attention_scores / (K.size(-1) ** 0.5)  # 缩放
        
        # 通过softmax获得注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, 40, 40)
        
        # 应用注意力权重到值上
        attended_values = torch.matmul(attention_weights, V)  # (batch_size, 40, hidden_dim)
        output = self.fc(attended_values) # (batch_size, 40, 13)
        return output
    
class RowAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RowAttention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        Q = self.query(x)  # (batch_size, 13, hidden_dim)
        K = self.key(x)    # (batch_size, 13, hidden_dim)
        V = self.value(x)  # (batch_size, 13, hidden_dim)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, 13, 13)
        attention_scores = attention_scores / (K.size(-1) ** 0.5)  # Scaling
        
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, 13, 13)
        
        attended_values = torch.matmul(attention_weights, V)  # (batch_size, 13, hidden_dim)
        output = self.fc(attended_values)  # (batch_size, 13, 40)
        return output

class Feed_Forward(nn.Module):
    def __init__(self, input_dim, d_ff, dropout_rate=0.1):
        super(Feed_Forward, self).__init__()
        # 第一个全连接层，输入维度为 input_dim，输出维度为 d_ff
        self.fc1 = nn.Linear(input_dim, d_ff)
        # 第二个全连接层，输入维度为 d_ff，输出维度为 input_dim
        self.fc2 = nn.Linear(d_ff, input_dim)
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout_rate)
        # 激活函数：ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        # 通过第一个全连接层
        x1 = self.fc1(x)
        # 激活函数
        x1 = self.relu(x1)
        # Dropout
        x1 = self.dropout(x1)
        # 通过第二个全连接层
        x2 = self.fc2(x1)
        # 残差连接：输入与输出相加
        out = x + x2  # x 是输入，x2 是第二个全连接层的输出
        return out

class CrossAttentionModel(nn.Module):
    def __init__(self, input_dim=13, seq_len=40, hidden_dim=64, output_dim=1, d_ff= 32, dropout_rate = 0.15):
        super(CrossAttentionModel, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.column_attention = ColumnAttention(input_dim, hidden_dim)
        self.row_attention = RowAttention(input_dim = 40, hidden_dim = 64)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.ffn1 = Feed_Forward(input_dim, d_ff, dropout_rate)
        self.ffn2 = Feed_Forward(input_dim, d_ff, dropout_rate)
        # 全连接层
        self.fc1 = nn.Linear(input_dim * seq_len, 128)
        self.fc2 = nn.Linear(128, output_dim)
        # self.sigmoid = nn.Sigmoid()
        self.droupout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 输入 x 维度: (batch_size, 40, 13)
        x = x.view(-1, self.seq_len, self.input_dim)  # 调整输入形状
        x = self.column_attention(x)
        x = self.layer_norm1(x)  # Layer Norm after Column Attention
        x = self.ffn1(x)  # Feed-Forward Neural Network
        # return x
        # Step 2: Row Attention
        x = x.transpose(1, 2)  # Change shape to (batch_size, 13, 40) for row attention
        # return x
        x = self.row_attention(x)

        x = x.transpose(1, 2)
        # return x
        x = self.layer_norm2(x)  # Layer Norm after Row Attention
        x = self.ffn2(x)  # Feed-Forward Neural Network
        
        # 将输出展平
        attention_output = x.view(-1, self.seq_len * self.input_dim)
        
        # 全连接层
        x = F.relu(self.fc1(attention_output))  # (batch_size, 128)
        x = self.droupout(x) # (batch_size, 128)
        x = self.fc2(x)  # (batch_size, output_dim = 1)

        return x
    
    # def _initialize_weights(self):
    #     # 对column_attention的linear层进行初始化
    #     init.kaiming_normal_(self.column_attention.query.weight, nonlinearity='relu')
    #     init.kaiming_normal_(self.column_attention.key.weight, nonlinearity='relu')
    #     init.kaiming_normal_(self.column_attention.value.weight, nonlinearity='relu')
        
    #     # 对fc1和fc2进行初始化
    #     init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
    #     init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        
    #     # 初始化偏置项为0
    #     if self.column_attention.query.bias is not None:
    #         init.constant_(self.column_attention.query.bias, 0)
    #         init.constant_(self.column_attention.key.bias, 0)
    #         init.constant_(self.column_attention.value.bias, 0)
        
    #     if self.fc1.bias is not None:
    #         init.constant_(self.fc1.bias, 0)
    #     if self.fc2.bias is not None:
    #         init.constant_(self.fc2.bias, 0)
    def _initialize_weights(self):
        # 对 column_attention 的 linear 层进行初始化
        init.kaiming_normal_(self.column_attention.query.weight, nonlinearity='relu')
        init.kaiming_normal_(self.column_attention.key.weight, nonlinearity='relu')
        init.kaiming_normal_(self.column_attention.value.weight, nonlinearity='relu')
        
        # 对 row_attention 的 linear 层进行初始化
        init.kaiming_normal_(self.row_attention.query.weight, nonlinearity='relu')
        init.kaiming_normal_(self.row_attention.key.weight, nonlinearity='relu')
        init.kaiming_normal_(self.row_attention.value.weight, nonlinearity='relu')
        
        # 对 ffn1 和 ffn2 的 linear 层进行初始化
        init.kaiming_normal_(self.ffn1.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.ffn1.fc2.weight, nonlinearity='relu')
        init.kaiming_normal_(self.ffn2.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.ffn2.fc2.weight, nonlinearity='relu')
        
        # 对 fc1 和 fc2 进行初始化
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        
        # 初始化偏置项为 0
        if self.column_attention.query.bias is not None:
            init.constant_(self.column_attention.query.bias, 0)
            init.constant_(self.column_attention.key.bias, 0)
            init.constant_(self.column_attention.value.bias, 0)
        
        if self.row_attention.query.bias is not None:
            init.constant_(self.row_attention.query.bias, 0)
            init.constant_(self.row_attention.key.bias, 0)
            init.constant_(self.row_attention.value.bias, 0)
        
        if self.ffn1.fc1.bias is not None:
            init.constant_(self.ffn1.fc1.bias, 0)
            init.constant_(self.ffn1.fc2.bias, 0)
        
        if self.ffn2.fc1.bias is not None:
            init.constant_(self.ffn2.fc1.bias, 0)
            init.constant_(self.ffn2.fc2.bias, 0)
        
        if self.fc1.bias is not None:
            init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            init.constant_(self.fc2.bias, 0)
                
class MultiColumnAttentionModel(nn.Module):
    def __init__(self, input_dim=16, seq_len=40, hidden_dim=64, output_dim=1, d_ff= 32, dropout_rate = 0.15):
        super(CrossAttentionModel, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.column_attention = ColumnAttention(input_dim, hidden_dim)
        self.row_attention = RowAttention(input_dim = 40, hidden_dim = 64)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.ffn1 = Feed_Forward(input_dim, d_ff, dropout_rate)
        self.ffn2 = Feed_Forward(input_dim, d_ff, dropout_rate)
        # 全连接层
        self.fc1 = nn.Linear(input_dim * seq_len, 128)
        self.fc2 = nn.Linear(128, output_dim)
        # self.sigmoid = nn.Sigmoid()
        self.droupout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 输入 x 维度: (batch_size, 40, 16)
        x = x.view(-1, self.seq_len, self.input_dim)  # 调整输入形状

        x = self.column_attention(x)
        x = self.layer_norm1(x)  # Layer Norm after Column Attention
        x = self.ffn1(x)  # Feed-Forward Neural Network,(batch_size, 16, 40)

        x = self.column_attention(x)
        x = self.layer_norm2(x)  # Layer Norm after Row Attention
        x = self.ffn2(x)  # Feed-Forward Neural Network,(batch_size, 16, 40)
        
        # 将输出展平
        attention_output = x.view(-1, self.seq_len * self.input_dim)
        
        # 全连接层
        x = F.relu(self.fc1(attention_output))  # (batch_size, 128)
        x = self.droupout(x) # (batch_size, 128)
        x = self.fc2(x)  # (batch_size, output_dim = 1)

        return x
    
    def _initialize_weights(self):
        # 对column_attention的linear层进行初始化
        init.kaiming_normal_(self.column_attention.query.weight, nonlinearity='relu')
        init.kaiming_normal_(self.column_attention.key.weight, nonlinearity='relu')
        init.kaiming_normal_(self.column_attention.value.weight, nonlinearity='relu')
        
        # 对fc1和fc2进行初始化
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        
        # 初始化偏置项为0
        if self.column_attention.query.bias is not None:
            init.constant_(self.column_attention.query.bias, 0)
            init.constant_(self.column_attention.key.bias, 0)
            init.constant_(self.column_attention.value.bias, 0)
        
        if self.fc1.bias is not None:
            init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            init.constant_(self.fc2.bias, 0) 


# %%
class CustomDataset(Dataset):
    def __init__(self, features):
        self.features = features
        # self.labels = labels
        self.position_encoding = self.create_position_encoding()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 将每个样本的 520 维特征转换为 40x13 的二维张量
        sample = self.features[idx].reshape(16, 40).astype(np.float32).T
        # label = torch.tensor(self.labels[idx], dtype=torch.long)
        position_encoding = self.position_encoding.numpy()  # shape (seq_len, input_dim)
        # sample_with_pos = sample + self.position_encoding
        return torch.tensor(sample)
    
    def create_position_encoding(self, seq_len = 40, input_dim = 16):
        # 获取位置序列，shape: (seq_len, 1)
        position = np.arange(0, seq_len).reshape(-1, 1)  # 生成一个形状为 (seq_len, 1) 的位置向量 [0, 1, 2, ..., seq_len-1]

        # 计算不同维度的扩展因子，shape: (input_dim // 2,)
        div_term = np.exp(np.arange(0, input_dim // 2) * -(np.log(10000.0) / input_dim))  # 计算每个位置特征的频率调整因子

        # 创建一个形状为 (seq_len, input_dim) 的位置编码矩阵
        pos_enc = np.zeros((seq_len, input_dim))  # 初始化一个零矩阵

        # 对偶数位置使用 sin 函数，计算每个位置的偶数维度的值
        pos_enc[:, 0::2] = np.sin(position * div_term)  # 偶数位置使用正弦函数
        
        # 对奇数位置使用 cos 函数，计算每个位置的奇数维度的值
        pos_enc[:, 1::2] = np.cos(position * div_term)  # 奇数位置使用余弦函数

        return torch.tensor(pos_enc, dtype=torch.float32)  # 将生成的 numpy 数组转换为 torch tensor


# %%
# 加载模型
model_path = '/BioII/lulab_b/huangkeyun/zhangys/alkb-seq/ML_models/eight_sample_11features_test/DL_saved/11f_test0.2_nostop_60epoches_0.15drop_nopos/best_model_epoch_60_auc_0.860.pth'
model = CrossAttentionModel(input_dim=16, seq_len=40, hidden_dim=64, dropout_rate=0.15)
model.load_state_dict(torch.load(model_path))
model.eval()
print("Model loaded successfully!")
# 设置文件夹路径
input_folder_path = '/BioII/lulab_b/huangkeyun/zhangys/alkb-seq/predict_TCGA/output/SelfAttentionSamples/samples/'
output_folder_path = '/BioII/lulab_b/huangkeyun/zhangys/alkb-seq/predict_TCGA/prediction_ML/saved_prediction/'

# 获取该文件夹下所有 CSV 文件的文件名，并按文件名顺序排序
csv_files = sorted([f for f in os.listdir(input_folder_path) if f.endswith('.csv')])

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder_path, exist_ok=True)

# 依次读取每个 CSV 文件并进行预测
for idx, file in enumerate(csv_files):
    input_file_path = os.path.join(input_folder_path, file)
    output_file_path = os.path.join(output_folder_path, file)

    # 检查是否已有输出文件
    if os.path.exists(output_file_path):
        print(f"文件 {file} 已存在，跳过预测。")
        continue

    # 读取 CSV 文件
    df = pd.read_csv(input_file_path)
    features = df.iloc[:, 1:].values  # 第一列为 'sample'

    # 创建数据集和数据加载器
    dataset = CustomDataset(features)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 进行预测
    predictions = []
    with torch.no_grad():
        for inputs in dataloader:
            outputs = model(inputs)
            predictions.extend(outputs.numpy())

    # 将预测结果保存到 CSV 文件中
    sample_names = df.iloc[:, 0].values  # 获取样本名
    predictions = np.array(predictions).flatten()  # 将预测结果展平为一维数组
    predicted_labels = (predictions > 0.5).astype(int)  # 预测标签，>0.5 标记为 1，否则标记为 0
    results_df = pd.DataFrame({'sample': sample_names, 'prediction': predictions, 'predicted_label': predicted_labels})
    results_df.to_csv(output_file_path, index=False)

    # 报数进行进度监控
    print(f"已完成 {idx + 1}/{len(csv_files)} 个文件的预测。")

