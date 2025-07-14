import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import gc

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("pic_freq2", exist_ok=True)

# ========== 数据预处理 ==========
def remove_outliers_zscore(df, cols, z_thresh=3):
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        z_scores = (series - series.mean()) / (series.std() + 1e-8)
        outliers = np.abs(z_scores) > z_thresh
        df.loc[outliers, col] = np.nan
    return df

def load_and_process(filepath, has_header=True):
    cols = ['DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
    df = pd.read_csv(filepath, header=0 if has_header else None, names=cols, parse_dates=['DateTime'], dtype=str)

    for col in cols[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    df['energy_kWh'] = df['Global_active_power'] * (1.0 / 60.0)
    df['RR'] = df['RR'] / 10.0
    df['date'] = df['DateTime'].dt.date

    numeric_cols = ['energy_kWh', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
                    'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
    df = remove_outliers_zscore(df, numeric_cols, z_thresh=2)
    df.interpolate(method='linear', inplace=True, limit_direction='both')

    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    daily = df.groupby('date').agg({
        'energy_kWh': 'sum',
        'Global_reactive_power': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'RR': 'first',
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first'
    }).reset_index()

    daily['sub_metering_remainder'] = (daily['energy_kWh'] * 1000) - (
        daily['Sub_metering_1'] + daily['Sub_metering_2'] + daily['Sub_metering_3'])

    return daily

# ========== Dataset ==========
class ElectricDataset(Dataset):
    def __init__(self, df, input_len=90, pred_len=90):
        self.features = df.columns.drop('date')
        self.values = df[self.features].values.astype(np.float32)
        self.min = self.values.min(axis=0)
        self.max = self.values.max(axis=0)
        self.values = (self.values - self.min) / (self.max - self.min + 1e-8)
        self.input_len = input_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.values) - self.input_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.values[idx: idx + self.input_len]
        y = self.values[idx + self.input_len: idx + self.input_len + self.pred_len, 0]
        return torch.tensor(x), torch.tensor(y)

# ========== 模型 ==========
class FreqFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ffted = torch.fft.fft(x, dim=1)
        real = ffted.real
        imag = ffted.imag
        return torch.cat([real, imag], dim=-1)

class LSTMPredictorWithFreq(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, output_len=90, dropout=0.2):
        super().__init__()
        self.freq_extractor = FreqFeatureExtractor()
        self.lstm = nn.LSTM(input_size * 2, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_len)

    def forward(self, x):
        freq_feat = self.freq_extractor(x)
        out, _ = self.lstm(freq_feat)
        return self.fc(out[:, -1, :])

# ========== 训练与评估 ==========
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, min_val, max_val, return_series=False):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred = pred.cpu().numpy() * (max_val[0] - min_val[0] + 1e-8) + min_val[0]
            y = y.cpu().numpy() * (max_val[0] - min_val[0] + 1e-8) + min_val[0]
            preds.append(pred)
            trues.append(y)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    if return_series:
        return mse, mae, trues.reshape(-1, pred.shape[1]), preds.reshape(-1, pred.shape[1])
    return mse, mae

def get_loss_fn(loss_type="smoothl1"):
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "mae":
        return nn.L1Loss()
    elif loss_type == "smoothl1":
        return nn.SmoothL1Loss()
    elif loss_type == "combined":
        mse = nn.MSELoss()
        mae = nn.L1Loss()
        def combined_loss(pred, target):
            return 0.5 * mse(pred, target) + 0.5 * mae(pred, target)
        return combined_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# ========== 可视化 ==========
def plot_predictions(trues, preds, run_idx, indices=[0]):
    for i in indices:
        plt.figure(figsize=(12, 4))
        plt.plot(trues[i], label='True', linewidth=2)
        plt.plot(preds[i], '--', label='Predicted', linewidth=2)
        plt.title(f'Run {run_idx+1} Sample {i} - Predicted vs Ground Truth')
        plt.xlabel('Day')
        plt.ylabel('Energy Consumption (kWh)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"pic_freq2/run{run_idx+1}_sample{i+1}_prediction.png", dpi=300)
        plt.close()

def plot_loss_curve(losses, run_idx):
    plt.figure()
    plt.plot(losses, label="Loss")
    plt.title(f"Run {run_idx+1} - Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"pic_freq2/run{run_idx+1}_loss_curve.png", dpi=300)
    plt.close()

# ========== 主函数 ==========
def main(train_path, test_path, input_len=90, pred_len=90, epochs=200, batch_size=32, loss_type="combined"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Loss Function: {loss_type}")

    train_df = load_and_process(train_path, has_header=True)
    test_df = load_and_process(test_path, has_header=False)

    train_data = ElectricDataset(train_df, input_len, pred_len)
    test_data = ElectricDataset(test_df, input_len, pred_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    mse_list, mae_list = [] , []
    input_size = len(train_data.features)

    for run in range(5):
        model = LSTMPredictorWithFreq(input_size=input_size, output_len=pred_len, dropout=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = get_loss_fn(loss_type)
        losses = []

        for epoch in range(epochs):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            losses.append(loss)
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'Run {run+1} - Epoch {epoch+1}: Loss = {loss:.4f}')

        plot_loss_curve(losses, run)
        mse, mae, trues, preds = evaluate(model, test_loader, device, train_data.min, train_data.max, return_series=True)
        plot_predictions(trues, preds, run, indices=[0])
        mse_list.append(mse)
        mae_list.append(mae)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    print("\n===  Final Evaluation over 5 Runs ===")
    for i in range(5):
        print(f'Run {i + 1}: MSE = {mse_list[i]:.4f}, MAE = {mae_list[i]:.4f}')
    print(f"\nMSE Mean: {np.mean(mse_list):.4f}, Std: {np.std(mse_list):.4f}")
    print(f"MAE Mean: {np.mean(mae_list):.4f}, Std: {np.std(mae_list):.4f}")


if __name__ == '__main__':
    train_csv = './datasets/train.csv'
    test_csv = './datasets/test.csv'
    main(train_csv, test_csv)
