import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from mpl_toolkits.axes_grid1.inset_locator import inset_axes # 用于画局部放大图
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心 SCN 类 (支持更多节点)
# ==========================================
class SimpleSCN:
    def __init__(self, n_hidden_nodes=100, regularization=0.1):
        self.L = n_hidden_nodes
        self.C = regularization
        self.W = None
        self.b = None
        self.beta = None

    def fit(self, X, y):
        # 1. 随机生成输入权重和偏置
        input_dim = X.shape[1]
        np.random.seed(None) # 确保完全随机
        self.W = np.random.uniform(-1, 1, (input_dim, self.L))
        self.b = np.random.uniform(-1, 1, (self.L))
        
        # 2. 计算隐含层矩阵 H
        H = self._activation(np.dot(X, self.W) + self.b)
        
        # 3. 岭回归求解输出权重 beta
        # beta = (H.T * H + C * I)^-1 * H.T * y
        identity = np.eye(self.L)
        # 使用伪逆求解更稳定，防止矩阵奇异
        # 这里对应论文中的最小二乘法求解
        H_inv = np.linalg.inv(np.dot(H.T, H) + self.C * identity)
        self.beta = np.dot(np.dot(H_inv, H.T), y)

    def predict(self, X):
        H = self._activation(np.dot(X, self.W) + self.b)
        return np.dot(H, self.beta)

    def _activation(self, x):
        # Sigmoid 激活函数
        return 1 / (1 + np.exp(-x))

# ==========================================
# 2. 数据加载与预处理
# ==========================================
def load_data(filepath='Folds5x2_pp.xlsx'):
    try:
        print(f"正在读取数据: {filepath} ...")
        df = pd.read_excel(filepath)
        X = df[['AT', 'V', 'AP', 'RH']].values
        y = df['PE'].values
    except FileNotFoundError:
        print(">>> 警告: 未找到 'Folds5x2_pp.xlsx'。生成模拟非线性数据用于演示。")
        # 生成模拟数据 (模拟 CCPP 的非线性关系)
        N = 2000
        X = np.random.rand(N, 4)
        # y = 10*sin(x0) + x1^2 + ...
        y = 10 * np.sin(np.pi * X[:, 0]) + 5 * (X[:, 1]**2) - 2 * X[:, 2] + 10 + np.random.normal(0, 0.1, N)
    
    # 全局归一化
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # 划分数据集 (80% 训练, 20% 测试)
    return train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42), scaler_y

# ==========================================
# 3. 评估工具
# ==========================================
def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# ==========================================
# 4. 主程序：改进版 DWMP-SCN
# ==========================================
if __name__ == "__main__":
    # --- 配置参数 (关键改进点) ---
    SCN_NODES = 500         # [改进2] 暴力增加节点数 (原100 -> 500)
    POOL_SIZE = 50          # 模型池大小
    TOP_K = 10              # 选出最好的10个
    SOFTMAX_LAMBDA = 50     # [改进3] Softmax 系数 (越大，好模型权重越高)
    BAGGING_RATIO = 0.8     # [改进1] Bagging 采样比例
    
    # 1. 加载数据
    (X_train, X_test, y_train, y_test), scaler_y = load_data()
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    
    # 2. 训练基准模型 (Standard SCN) 用于对比
    print("-" * 50)
    print("正在训练 Standard SCN (基准)...")
    base_scn = SimpleSCN(n_hidden_nodes=SCN_NODES)
    base_scn.fit(X_train, y_train)
    y_pred_base = base_scn.predict(X_test)
    rmse_base, _, _ = evaluate(y_test, y_pred_base, "Base")
    print(f"Standard SCN RMSE: {rmse_base:.5f}")

    # 3. 训练改进版 DWMP-SCN
    print("-" * 50)
    print(f"正在构建改进版 DWMP-SCN (Nodes={SCN_NODES}, Bagging=ON, Softmax=ON)...")
    
    model_pool = []
    
    for i in range(POOL_SIZE):
        # [改进1] Bagging: 随机抽取 80% 的训练数据
        # 使用 train_test_split 实现随机采样
        X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=BAGGING_RATIO, random_state=None)
        
        # 训练子模型
        model = SimpleSCN(n_hidden_nodes=SCN_NODES)
        model.fit(X_sub, y_sub)
        
        # 在*完整*训练集上评估误差 (用于筛选)
        train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        
        model_pool.append({'model': model, 'rmse': train_rmse})
        
        if (i+1) % 10 == 0:
            print(f"  已训练 {i+1}/{POOL_SIZE} 个子模型...")

    # 4. 筛选 (Selection)
    model_pool.sort(key=lambda x: x['rmse'])
    best_models = model_pool[:TOP_K]
    
    # 5. [改进3] Softmax 加权 (Weighting)
    # 提取 RMSE 列表
    rmses = np.array([m['rmse'] for m in best_models])
    # 计算 Softmax: w_i = exp(-lambda * rmse_i) / sum(...)
    # 减去最小值防止溢出 (数值稳定性技巧)
    exp_weights = np.exp(-SOFTMAX_LAMBDA * (rmses - np.min(rmses)))
    weights = exp_weights / np.sum(exp_weights)
    
    print(f"\nTop {TOP_K} 模型 RMSE: {[f'{r:.4f}' for r in rmses]}")
    print(f"Softmax 计算权重: {[f'{w:.3f}' for w in weights]}")

    # 6. 集成预测 (Ensemble Prediction)
    final_pred = np.zeros_like(y_test)
    for i, item in enumerate(best_models):
        pred = item['model'].predict(X_test)
        final_pred += weights[i] * pred
    
    rmse_dwmp, mae_dwmp, r2_dwmp = evaluate(y_test, final_pred, "DWMP-SCN")
    
    # 7. 打印最终填空数据
    improvement = (rmse_base - rmse_dwmp) / rmse_base * 100
    
    print("=" * 50)
    print(">>> 【论文填空数据】 <<<")
    print(f"Standard SCN RMSE (对比): {rmse_base:.5f}")
    print(f"DWMP-SCN RMSE     (本文): {rmse_dwmp:.5f}")
    print(f"RMSE 提升百分比:    {improvement:.2f}% (填入 Abstract)")
    print(f"R^2 Score:        {r2_dwmp:.5f} (填入 Abstract)")
    print("=" * 50)

    # ==========================================
    # 8. 专业绘图 (Papers Ready)
    # ==========================================
    # 还原真实量纲 (MW)
    y_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    pred_real = scaler_y.inverse_transform(final_pred.reshape(-1, 1)).ravel()
    base_real = scaler_y.inverse_transform(y_pred_base.reshape(-1, 1)).ravel()

    # --- 图1: 拟合曲线 + 局部放大 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    subset = 150 # 只画前150个点
    x_axis = np.arange(subset)
    
    ax.plot(x_axis, y_real[:subset], color='black', linewidth=1.5, label='Actual Value')
    ax.plot(x_axis, base_real[:subset], color='gray', linestyle=':', alpha=0.6, label='Standard SCN')
    ax.plot(x_axis, pred_real[:subset], color='red', linestyle='--', linewidth=1.5, label='DWMP-SCN (Ours)')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Power Output (MW)')
    ax.set_title('Fig. 1: Prediction Performance Comparison')
    ax.legend(loc='upper right')
    
    # 局部放大图
    axins = inset_axes(ax, width="35%", height="30%", loc='lower left', borderpad=3)
    zoom_s, zoom_e = 50, 80
    axins.plot(x_axis[zoom_s:zoom_e], y_real[zoom_s:zoom_e], 'k-', linewidth=1.5)
    axins.plot(x_axis[zoom_s:zoom_e], pred_real[zoom_s:zoom_e], 'r--', linewidth=1.5)
    axins.set_title("Zoom-in")
    axins.set_xticks([])
    axins.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    # --- 图2: 误差分布直方图 ---
    plt.figure(figsize=(8, 5))
    err_base = y_real - base_real
    err_our = y_real - pred_real
    
    plt.hist(err_base, bins=40, alpha=0.4, color='gray', label='Standard SCN Error', density=True)
    plt.hist(err_our, bins=40, alpha=0.8, color='red', label='DWMP-SCN Error', density=True)
    
    plt.xlabel('Prediction Error (MW)')
    plt.ylabel('Density')
    plt.title('Fig. 2: Error Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()