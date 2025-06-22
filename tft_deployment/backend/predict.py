import torch
import numpy as np

def make_prediction(model, input_sequence):
    x = torch.FloatTensor([input_sequence])  # shape (1, seq_len, num_features)
    with torch.no_grad():
        output = model(x)
    return output[0].numpy()
