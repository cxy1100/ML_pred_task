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

# 设置随机种子确保可重复
np.random.seed(42)
tf.random.set_seed(42)

# ------------------- 数据加载与处理 -------------------
print("加载数据并进行按天聚合...")
train_data = pd.read_csv('train.csv', parse_dates=['DateTime'])

def load_data(filepath):
    # 正确定义所有列名（必须与文件列顺序完全一致）
    column_names = [
        'DateTime',
        'global_active_power',  # 原错误可能使用了'Global_act'
        'global_reactive_power',  # 原错误可能使用了'Global_rec'
        'voltage',
        'global_intensity',  # 原错误可能使用了'Global_int'
        'sub_metering_1',
        'sub_metering_2',
        'sub_metering_3',
        'RR',
        'NBJRR1',
        'NBJRR5',
        'NBJRR10',
        'NBJBROU'
    ]

    # 加载时自动将非法字符转为NaN
    df = pd.read_csv(
        filepath,
        header=None,
        parse_dates=[0],
        names=column_names,
        na_values=['?', ' ', 'NA', 'N/A'],  # 指定所有可能的非法字符
        low_memory=False
    )

    # 强制转换数值列类型
    numeric_cols = column_names[1:8]  # 第2-8列应为数值
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # 处理可能的空值
    return df.dropna()


# 加载数据
test_data = load_data('test.csv')

train_data['Date'] = train_data['DateTime'].dt.date
test_data['Date'] = test_data['DateTime'].dt.date


def daily_aggregate(df):
    # 确保DateTime列正确
    if not pd.api.types.is_datetime64_any_dtype(df['DateTime']):
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

    # 创建日期列（保留datetime类型以便后续操作）
    df['Date'] = df['DateTime'].dt.floor('D')

    # 定义聚合字典
    agg_dict = {
        'global_active_power': 'sum',
        'global_reactive_power': 'sum',
        'voltage': 'mean',
        'global_intensity': 'mean',
        'sub_metering_1': 'sum',
        'sub_metering_2': 'sum',
        'sub_metering_3': 'sum',
        'RR': 'first',
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first'
    }

    # 仅聚合存在的列
    valid_cols = [col for col in agg_dict.keys() if col in df.columns]
    if not valid_cols:
        raise ValueError("数据中没有可聚合的列！请检查列名")

    df_daily = df.groupby('Date').agg({k: agg_dict[k] for k in valid_cols})

    # 5. 计算剩余能耗（确保所需列存在）
    required_cols = ['global_active_power', 'sub_metering_1', 'sub_metering_2', 'sub_metering_3']
    if all(col in df_daily.columns for col in required_cols):
        df_daily['Sub_metering_remainder'] = (df_daily['global_active_power'] * 1000 / 60) - (
                df_daily['sub_metering_1'] + df_daily['sub_metering_2'] + df_daily['sub_metering_3']
        )
    else:
        missing =[col for col in required_cols if col not in df_daily.columns]
        print(f"警告：缺少计算剩余能耗所需的列 {missing}，跳过计算")

    return df_daily.dropna()


train_daily = daily_aggregate(train_data)
test_daily = daily_aggregate(test_data)

# 特征
features = ['global_active_power', 'global_reactive_power', 'voltage', 'global_intensity',
            'sub_metering_1', 'sub_metering_2', 'sub_metering_3',
            'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU',
            'Sub_metering_remainder']

# 过滤掉不存在的特征
valid_features = [feature for feature in features if feature in train_daily.columns and feature in test_daily.columns]

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
        # 目标序列：未来pred_len天的global_active_power（特征列表中第一个特征）
        target = data[i + seq_len: i + seq_len + pred_len, valid_features.index('global_active_power')]
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
    x = Dense(forecast_len)(x)
    model = Model(inputs, x)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model


# ------------------- 多轮训练与评估 -------------------
def run_experiments(X_train, y_train, X_test, y_test, forecast_len, label):
    mse_list = []
    mae_list = []
    print(f"\n开始 {label} 的 5 轮训练与评估:")

    for i in range(5):
        print(f"\n第 {i + 1} 轮训练：")
        tf.keras.backend.clear_session()
        model = build_transformer((X_train.shape[1], X_train.shape[2]), forecast_len)
        # 检查y_train的形状
        if len(y_train.shape) == 3:
            y_train = y_train.reshape(y_train.shape[0], -1)
        if len(y_test.shape) == 3:
            y_test = y_test.reshape(y_test.shape[0], -1)
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        mse_list.append(mse)
        mae_list.append(mae)

        print(f"第 {i + 1} 轮结果 - MSE: {mse:.4f}, MAE: {mae:.4f}")

    mse_mean, mse_std = np.mean(mse_list), np.std(mse_list)
    mae_mean, mae_std = np.mean(mae_list), np.std(mae_list)

    print(f"\n[{label} 5轮评估汇总]")
    print(f"MSE 平均值: {mse_mean:.4f}, 标准差: {mse_std:.4f}")
    print(f"MAE 平均值: {mae_mean:.4f}, 标准差: {mae_std:.4f}")

    return mse_list, mae_list, mse_mean, mse_std, mae_mean, mae_std


# ------------------- 执行短期与长期预测 -------------------
short_results = run_experiments(X_train_90, y_train_90, X_test_90, y_test_90, 90, "短期预测（90天）")
long_results = run_experiments(X_train_365, y_train_365, X_test_365, y_test_365, 365, "长期预测（365天）")

# 保存短期结果
short_keys = ["mse_list", "mae_list", "mse_mean", "mse_std", "mae_mean", "mae_std"]
short_values = [*short_results]
short_dict = dict(zip(short_keys, short_values))
with open("short_term_results.json", "w") as f:
    json.dump(short_dict, f, indent=4)

# 保存长期结果
long_keys = ["mse_list", "mae_list", "mse_mean", "mse_std", "mae_mean", "mae_std"]
long_values = [*long_results]
long_dict = dict(zip(long_keys, long_values))
with open("long_term_results.json", "w") as f:
    json.dump(long_dict, f, indent=4)

print(" 所有指标已保存至 JSON 文件：short_term_results.json 和 long_term_results.json")