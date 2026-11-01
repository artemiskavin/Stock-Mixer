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

class NoGraphMixer(nn.Module):
    def __init__(self, stocks, hidden_dim=20):
        super(NoGraphMixer, self).__init__()
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, stocks)
        self.layer_norm_stock = nn.LayerNorm(stocks)

    def forward(self, inputs):
        x = inputs
        x = x.permute(1, 0)
        x = self.layer_norm_stock(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = x.permute(1, 0)
        return x

class LSTMStockForecaster(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, stocks=1026, market=20):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
        self.stock_mixer = NoGraphMixer(stocks, market)
        self.stock_head = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        sequence, _ = self.rnn(inputs)
        representation = sequence[:, -1, :]
        market_representation = self.stock_mixer(representation)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return self.head(representation.float()) + self.stock_head(market_representation.float())
