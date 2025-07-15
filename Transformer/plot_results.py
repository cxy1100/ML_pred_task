# plot_results.py
import matplotlib.pyplot as plt
import numpy as np
import json

def summarize_and_plot(metric_list, mean_val, std_val, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(1, len(metric_list)+1), metric_list, color='skyblue', edgecolor='black', width=0.4)

    # 平均线与标准差带
    plt.axhline(mean_val, color='red', linestyle='--', label=f'average value: {mean_val:.4f}')
    plt.axhline(mean_val + std_val, color='orange', linestyle=':', label=f'±1σ: {std_val:.4f}')
    plt.axhline(mean_val - std_val, color='orange', linestyle=':')

    for idx, val in enumerate(metric_list):
        plt.text(idx + 1, val + 0.001, f"{val:.4f}", ha='center', fontsize=12)

    plt.title(title, fontsize=16, pad=20)   # 调整标题的字体大小和向上移动的距离
    plt.xlabel("Experimental rounds", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(range(1, len(metric_list)+1), fontsize=10)  # 调整x轴刻度标签的字体大小
    plt.legend(fontsize=10)  # 调整图例的字体大小
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(pad=1)  # 调整边界框的填充空间，使边界框变宽
    plt.savefig(filename)
    plt.show()

# 修改后的绘制损失折线图的函数
def plot_loss(loss_list, title, ylabel, filename):
    plt.figure(figsize=(12, 8))

    # 绘制每一轮训练的损失曲线
    colors = plt.cm.tab10.colors  # 使用10种不同颜色区分轮次
    for i, losses in enumerate(loss_list):
        color = colors[i % len(colors)]
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, marker='o', linestyle='-', color=color, label=f'Round {i + 1}')

        # 在最后一个点上标注最终损失值
        plt.annotate(f'{losses[-1]:.4f}',
                     xy=(len(losses), losses[-1]),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=10)

    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Training Rounds', fontsize=10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# ------------------- 从JSON中读取 -------------------
def load_results(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

short_data = load_results("short_term_results3.json")
long_data = load_results("long_term_results3.json")

# ------------------- 绘制图 -------------------
summarize_and_plot(short_data["mse_list"], short_data["mse_mean"], short_data["mse_std"],
                   "short-term MSE 5 runs", "MSE", "mse_short_term3.png")

summarize_and_plot(short_data["mae_list"], short_data["mae_mean"], short_data["mae_std"],
                   "short-term MAE 5 runs", "MAE", "mae_short_term3.png")

summarize_and_plot(long_data["mse_list"], long_data["mse_mean"], long_data["mse_std"],
                   "long-term MSE 5 runs", "MSE", "mse_long_term3.png")

summarize_and_plot(long_data["mae_list"], long_data["mae_mean"], long_data["mae_std"],
                   "long-term MAE 5 runs", "MAE", "mae_long_term3.png")

# ------------------- 绘制电量预测与真实值（Ground Truth）曲线 -------------------

# 从 JSON 中读取预测和真实值
y_test_90 = short_data['y_test_list']  # 短期预测的 y_test 存储在 short_term_results1.json 中
y_pred_90 = short_data['y_pred_list']  # 短期预测的 y_pred 存储在 short_term_results1.json 中

y_test_365 = long_data['y_test_list']  # 长期预测
y_pred_365 = long_data['y_pred_list']

# 短期预测（90天）对比图，只选取第一个样本
plt.figure(figsize=(10, 6))
plt.plot(y_test_90[0][0], label='Ground Truth', color='blue')
plt.plot(y_pred_90[0][0], label='Prediction', color='red', linestyle='--')
plt.title("Short-term Energy Consumption 90 days", fontsize=14)
plt.xlabel("days", fontsize=12)
plt.ylabel("Energy Consumption", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("power_comparison_short_term3.png")
plt.show()

# 长期预测（365天）对比图
plt.figure(figsize=(10, 6))
plt.plot(y_test_365[0][0], label='Ground Truth', color='blue')
plt.plot(y_pred_365[0][0], label='Prediction', color='red', linestyle='--')
plt.title("Energy Consumption 365 days", fontsize=14)
plt.xlabel("days", fontsize=12)
plt.ylabel("Energy Consumption", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("power_comparison_long_term3.png")
plt.show()

# ------------------- 绘制损失折线图 -------------------
plot_loss(short_data['loss_list'], "Short-term Loss per Epoch", "Loss", "short_term_loss3.png")

plot_loss(long_data['loss_list'], "Long-term Loss per Epoch", "Loss", "long_term_loss3.png")