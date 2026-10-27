"""Row-column attention architecture; returns unthresholded logits."""

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


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
        
        self.fc1 = nn.Linear(input_dim * seq_len, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.droupout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Input shape adjustment
        x = x.reshape(-1, self.seq_len, self.input_dim)
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
        attention_output = x.reshape(-1, self.seq_len * self.input_dim)
        
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
