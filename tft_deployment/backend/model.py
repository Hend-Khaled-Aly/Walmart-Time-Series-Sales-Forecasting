import torch
from model_def import TFTInspiredLSTM  # Same class you defined
import os

def load_model():
    input_size = 6  # adjust to your final number of features
    hidden_size = 64
    num_layers = 2
    output_size = 12  # prediction length

    model = TFTInspiredLSTM(input_size, hidden_size, num_layers, output_size)
    model.load_state_dict(torch.load("tft_lstm_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model
