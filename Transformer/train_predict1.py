# train_predict.py
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

np.random.seed(42)
tf.random.set_seed(42)

# 统一列名映射（与 train.csv 中一致）
column_names = [
    'DateTime',
    'Global_active_power',
    'Global_reactive_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3',
    'RR',
    'NBJRR1',
    'NBJRR5',
    'NBJRR10',
    'NBJBROU'
]

# 封装统一加载函数
def load_data(filepath, has_header=True):
    df = pd.read_csv(
        filepath,
        header=0 if has_header else None,
        names=column_names if not has_header else None,
        parse_dates=['DateTime'],
        na_values=['?', ' ', 'NA', 'N/A'],
        low_memory=False
    )
    numeric_cols = column_names[1:]  # 除了 DateTime，其它都是数值
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    return df.dropna()

# 使用统一加载函数（train 有列名，test 无列名）
print("加载数据并进行按天聚合...")
train_data = load_data('train.csv', has_header=True)
test_data = load_data('test.csv', has_header=False)

train_data['Date'] = train_data['DateTime'].dt.date
test_data['Date'] = test_data['DateTime'].dt.date

# 按天聚合
def daily_aggregate(df):
    df['Date'] = pd.to_datetime(df['DateTime']).dt.floor('D')

    # 转换 Global_active_power 到 energy_kWh
    df['energy_kWh'] = df['Global_active_power'] * (1.0 / 60.0)

    agg_dict = {
        'energy_kWh': 'sum',  # 修改这里
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
    }

    df_daily = df.groupby('Date').agg(agg_dict)

    # 补充计算剩余能耗，修改成energy_kWh
    required_cols = ['energy_kWh', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    if all(col in df_daily.columns for col in required_cols):    # 修改这里energy_kWh，已除过60
        df_daily['Sub_metering_remainder'] = (df_daily['energy_kWh'] * 1000) - (
            df_daily['Sub_metering_1'] + df_daily['Sub_metering_2'] + df_daily['Sub_metering_3']
        )
    else:
        print("⚠️ 缺失列，无法计算剩余能耗")

    # 处理其他列的数据
    df_daily['RR'] = df_daily['RR'] / 10.0

    return df_daily.dropna()

# 聚合后数据
train_daily = daily_aggregate(train_data)
test_daily = daily_aggregate(test_data)

# 特征一致性
features = [     # 修改成energy_kWh
    'energy_kWh', 'Global_reactive_power', 'Voltage', 'Global_intensity',
    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
    'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU',
    'Sub_metering_remainder'
]

valid_features = [f for f in features if f in train_daily.columns and f in test_daily.columns]

# 标准化，使用Min-Max标准化**：`y_test` 和 `y_pred` 的值在0到1之间
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_daily[valid_features])
test_scaled = scaler.transform(test_daily[valid_features])


# ------------------- 创建序列 -------------------
def create_sequences(data, seq_len, pred_len):
    """
     生成输入序列X和目标序列y
     X: 形状 (样本数, seq_len, 特征数)
     y: 形状 (样本数, pred_len)  （确保为二维）
     """
    X, y = [], []
    max_idx = len(data) - seq_len - pred_len + 1
    for i in range(max_idx):
        # 输入序列：过去seq_len天的所有特征
        X.append(data[i:i + seq_len, :])
        # 目标序列：未来pred_len天的global_active_power（特征列表中第一个特征），Global_active_power修改成energy_kWh？？？
        target = data[i + seq_len: i + seq_len + pred_len, valid_features.index('energy_kWh')]
        y.append(target)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    # 强制y为二维（防止多余维度）
    if len(y.shape) > 2:
        y = y.reshape(y.shape[0], -1)
    return X, y


# 生成短期（90天）和长期（365天）序列
seq_len = 90  # 用过去90天预测
X_train_90, y_train_90 = create_sequences(train_scaled, seq_len, 90)
X_test_90, y_test_90 = create_sequences(test_scaled, seq_len, 90)
X_train_365, y_train_365 = create_sequences(train_scaled, seq_len, 365)
X_test_365, y_test_365 = create_sequences(test_scaled, seq_len, 365)


# ------------------- Transformer模型 -------------------
def build_transformer(input_shape, forecast_len):
    inputs = Input(shape=input_shape)
    attn = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    x = LayerNormalization()(attn)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    # 对特征维度进行全局平均池化，将三维张量转换为二维张量
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    output = Dense(forecast_len)(x)  # output shape: (batch_size, forecast_len)
    model = Model(inputs, output)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model


# ------------------- 多轮训练与评估 -------------------
def run_experiments(X_train, y_train, X_test, y_test, forecast_len, label):
    mse_list = []
    mae_list = []
    y_pred_list = []
    y_test_list = []
    loss_list = []  # 新增：用于存储每一轮的损失值
    mse_original_list = []  # 新增：用于存储真实的mse值（未经过反标准化）
    mae_original_list = []  # 新增：用于存储真实的mae值（未经过反标准化）

    print(f"\n开始 {label} 的 5 轮训练与评估:")

    for i in range(5):
        print(f"\n第 {i + 1} 轮训练：")
        tf.keras.backend.clear_session()
        model = build_transformer((X_train.shape[1], X_train.shape[2]), forecast_len)

        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
        loss_list.append(history.history['loss'])  # 保存每一轮的损失值

        # model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
        # 预测
        y_pred = model.predict(X_test)

        # 在计算损失之前，转换 `y_pred` 和 `y_test` 的形状，此时`y_test` 和 `y_pred` 的值在0到1之间
        y_pred = y_pred.reshape(y_pred.shape[0], -1)  # 转换为二维数组 (batch_size, forecast_len)
        y_test = y_test.reshape(y_test.shape[0], -1)  # 转换为二维数组 (batch_size, forecast_len)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        mse_list.append(mse)
        mae_list.append(mae)

        # 反标准化，Global_active_power修改成energy_kWh？？
        global_active_power_index = valid_features.index('energy_kWh')
        min_val = scaler.data_min_[global_active_power_index]
        max_val = scaler.data_max_[global_active_power_index]
        y_pred_original = y_pred * (max_val - min_val) + min_val
        y_test_original = y_test * (max_val - min_val) + min_val

        # 计算真实值的mse和mae
        mse_original = mean_squared_error(y_test_original, y_pred_original)
        mae_original = mean_absolute_error(y_test_original, y_pred_original)

        # 将真实值的mse和mae写入列表存入文件
        mse_original_list.append(mse_original)
        mae_original_list.append(mae_original)

        y_pred_list.append(y_pred_original.tolist())
        y_test_list.append(y_test_original.tolist())

        print(f"第 {i + 1} 轮结果 - MSE: {mse:.4f}, MAE: {mae:.4f}")
        print(f"第 {i + 1} 轮结果 - MSE_Original: {mse_original:.4f}, MAE_Original: {mae_original:.4f}")

    mse_mean, mse_std = np.mean(mse_list), np.std(mse_list)
    mae_mean, mae_std = np.mean(mae_list), np.std(mae_list)

    # 计算真实值的mse_mean和mae_mean
    mse_mean_original, mse_std_original = np.mean(mse_original_list), np.std(mse_original_list)
    mae_mean_original, mae_std_original = np.mean(mae_original_list), np.std(mae_original_list)

    print(f"\n[{label} 5轮评估汇总]")
    print(f"MSE 平均值: {mse_mean:.4f}, 标准差: {mse_std:.4f}")
    print(f"MAE 平均值: {mae_mean:.4f}, 标准差: {mae_std:.4f}")

    print(f"MSE_Original 平均值: {mse_mean_original:.4f}, 标准差: {mse_std_original:.4f}")
    print(f"MAE_Original 平均值: {mae_mean_original:.4f}, 标准差: {mae_std_original:.4f}")

    return mse_list, mae_list, mse_mean, mse_std, mae_mean, mae_std, y_pred_list, y_test_list, loss_list, mse_original_list, mae_original_list


# ------------------- 执行短期与长期预测 -------------------
short_results = run_experiments(X_train_90, y_train_90, X_test_90, y_test_90, 90, "短期预测（90天）")
long_results = run_experiments(X_train_365, y_train_365, X_test_365, y_test_365, 365, "长期预测（365天）")

# 保存短期结果
short_keys = ["mse_list", "mae_list", "mse_mean", "mse_std", "mae_mean", "mae_std", "y_pred_list", "y_test_list", "loss_list", "mse_original_list", "mae_original_list"]
short_values = [*short_results]
short_dict = dict(zip(short_keys, short_values))
with open("short_term_results5.json", "w") as f:
    json.dump(short_dict, f, indent=4)

# 保存长期结果
long_keys = ["mse_list", "mae_list", "mse_mean", "mse_std", "mae_mean", "mae_std", "y_pred_list", "y_test_list", "loss_list", "mse_original_list", "mae_original_list"]
long_values = [*long_results]
long_dict = dict(zip(long_keys, long_values))
with open("long_term_results5.json", "w") as f:
    json.dump(long_dict, f, indent=4)

print(" 所有指标已保存至 JSON 文件：short_term_results5.json 和 long_term_results5.json")