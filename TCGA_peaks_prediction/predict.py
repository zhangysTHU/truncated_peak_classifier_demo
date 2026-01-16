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
        # Self-attention weights
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        # Input shape: (batch_size, 40, input_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Q * K^T scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        attention_scores = attention_scores / (K.size(-1) ** 0.5)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attended_values = torch.matmul(attention_weights, V)
        output = self.fc(attended_values) 
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
        self.fc1 = nn.Linear(input_dim, d_ff)
        self.fc2 = nn.Linear(d_ff, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x2 = self.fc2(x1)
        # Residual connection
        out = x + x2
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
        
        self.fc1 = nn.Linear(input_dim * seq_len, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.droupout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Input shape adjustment
        x = x.view(-1, self.seq_len, self.input_dim)
        x = self.column_attention(x)
        x = self.layer_norm1(x)
        x = self.ffn1(x)
        
        # Step 2: Row Attention
        x = x.transpose(1, 2)
        x = self.row_attention(x)
        x = x.transpose(1, 2)
        x = self.layer_norm2(x)
        x = self.ffn2(x)
        
        # Flatten for fully connected layers
        attention_output = x.view(-1, self.seq_len * self.input_dim)
        
        x = F.relu(self.fc1(attention_output))
        x = self.droupout(x)
        x = self.fc2(x)

        return x
    
    def _initialize_weights(self):
        # Initialize column_attention linear layers
        init.kaiming_normal_(self.column_attention.query.weight, nonlinearity='relu')
        init.kaiming_normal_(self.column_attention.key.weight, nonlinearity='relu')
        init.kaiming_normal_(self.column_attention.value.weight, nonlinearity='relu')
        
        # Initialize row_attention linear layers
        init.kaiming_normal_(self.row_attention.query.weight, nonlinearity='relu')
        init.kaiming_normal_(self.row_attention.key.weight, nonlinearity='relu')
        init.kaiming_normal_(self.row_attention.value.weight, nonlinearity='relu')
        
        # Initialize FFN linear layers
        init.kaiming_normal_(self.ffn1.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.ffn1.fc2.weight, nonlinearity='relu')
        init.kaiming_normal_(self.ffn2.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.ffn2.fc2.weight, nonlinearity='relu')
        
        # Initialize output heads
        init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        
        # Set biases to 0
        layers_with_bias = [
            self.column_attention.query, self.column_attention.key, self.column_attention.value,
            self.row_attention.query, self.row_attention.key, self.row_attention.value,
            self.ffn1.fc1, self.ffn1.fc2, self.ffn2.fc1, self.ffn2.fc2,
            self.fc1, self.fc2
        ]
        for layer in layers_with_bias:
            if layer.bias is not None:
                init.constant_(layer.bias, 0)
                
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
        x = x.view(-1, self.seq_len, self.input_dim)

        x = self.column_attention(x)
        x = self.layer_norm1(x)
        x = self.ffn1(x)

        x = self.column_attention(x)
        x = self.layer_norm2(x)
        x = self.ffn2(x)
        
        attention_output = x.view(-1, self.seq_len * self.input_dim)
        
        x = F.relu(self.fc1(attention_output))
        x = self.droupout(x)
        x = self.fc2(x)

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
        self.position_encoding = self.create_position_encoding()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Reshape flat features to (seq_len, input_dim)
        sample = self.features[idx].reshape(16, 40).astype(np.float32).T
        return torch.tensor(sample)
    
    def create_position_encoding(self, seq_len = 40, input_dim = 16):
        position = np.arange(0, seq_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, input_dim // 2) * -(np.log(10000.0) / input_dim))

        pos_enc = np.zeros((seq_len, input_dim))
        pos_enc[:, 0::2] = np.sin(position * div_term) # sine for even
        pos_enc[:, 1::2] = np.cos(position * div_term) # cosine for odd

        return torch.tensor(pos_enc, dtype=torch.float32)


# %%
# Load model
model_path = './ML_models/eight_sample_11features_test/DL_saved/11f_test0.2_nostop_60epoches_0.15drop_nopos/best_model_epoch_60_auc_0.860.pth'
model = CrossAttentionModel(input_dim=16, seq_len=40, hidden_dim=64, dropout_rate=0.15)
model.load_state_dict(torch.load(model_path))
model.eval()
print("Model loaded successfully!")

# Set IO paths
input_folder_path = './predict_TCGA/output/SelfAttentionSamples/samples/'
output_folder_path = './predict_TCGA/prediction_ML/saved_prediction/'

csv_files = sorted([f for f in os.listdir(input_folder_path) if f.endswith('.csv')])
os.makedirs(output_folder_path, exist_ok=True)

# Batch prediction
for idx, file in enumerate(csv_files):
    input_file_path = os.path.join(input_folder_path, file)
    output_file_path = os.path.join(output_folder_path, file)

    if os.path.exists(output_file_path):
        print(f"File {file} exists, skipping.")
        continue

    df = pd.read_csv(input_file_path)
    features = df.iloc[:, 1:].values # skip sample name column

    dataset = CustomDataset(features)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    predictions = []
    with torch.no_grad():
        for inputs in dataloader:
            outputs = model(inputs)
            predictions.extend(outputs.numpy())

    sample_names = df.iloc[:, 0].values
    predictions = np.array(predictions).flatten()
    predicted_labels = (predictions > 0.5).astype(int)
    results_df = pd.DataFrame({'sample': sample_names, 'prediction': predictions, 'predicted_label': predicted_labels})
    results_df.to_csv(output_file_path, index=False)

    print(f"Progress: {idx + 1}/{len(csv_files)}")

