"""Simple recurrent comparison. get_loss and NoGraphMixer copied with attribution
from Fan & Shen, AAAI 2024, SJTU-DMTai/StockMixer,
revision cce13598afd3ff33ae317700a85ae08db0554652. Behavior preserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t(),
        ground_truth @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=5, hidden_size=32):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        sequence, _ = self.rnn(inputs)
        representation = sequence[:, -1, :]
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return self.head(representation.float())
