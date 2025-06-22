import torch
import torch.nn as nn

# Define TFT-inspired LSTM model
class TFTInspiredLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
            super(TFTInspiredLSTM, self).__init__()
            
            # Multi-head attention simulation
            self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=2, dropout=dropout)
            
            # LSTM layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
            
            # Feed-forward network
            self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc2 = nn.Linear(hidden_size // 2, output_size)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # x shape: (batch_size, seq_len, features)
            
            # Apply attention (simplified)
            x_att = x.transpose(0, 1)  # (seq_len, batch_size, features)
            attn_output, _ = self.attention(x_att, x_att, x_att)
            x_att = attn_output.transpose(0, 1)  # Back to (batch_size, seq_len, features)
            
            # Combine original and attention
            x_combined = x + x_att
            
            # LSTM processing
            lstm_out, _ = self.lstm(x_combined)
            
            # Use last output for prediction
            last_output = lstm_out[:, -1, :]
            
            # Feed-forward layers
            x = self.relu(self.fc1(last_output))
            x = self.dropout(x)
            output = self.fc2(x)
            
            return output