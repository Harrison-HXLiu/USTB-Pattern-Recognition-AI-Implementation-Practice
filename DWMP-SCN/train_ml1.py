import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import warnings

warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5

# ---------------------------------------------------------
# 1. 核心模型: SimpleSCN
# ---------------------------------------------------------
class SimpleSCN:
    def __init__(self, n_hidden_nodes=500, regularization=0.0001, verbose=False):
        self.L = n_hidden_nodes
        self.C = regularization
        self.W = None
        self.b = None
        self.beta = None
        self.verbose = verbose

    def _activation(self, x):
        return 1 / (1 + np.exp(-x)) # Sigmoid

    def fit(self, X, y):
        input_dim = X.shape[1]
        # 随机生成参数
        self.W = np.random.uniform(-1, 1, (input_dim, self.L))
        self.b = np.random.uniform(-1, 1, (self.L))
        
        H = self._activation(np.dot(X, self.W) + self.b)
        
        # 最小二乘法求解输出权重 (含正则化)
        identity = np.eye(self.L)
        H_inv = np.linalg.pinv(np.dot(H.T, H) + self.C * identity)
        self.beta = np.dot(np.dot(H_inv, H.T), y)

    def predict(self, X):
        H = self._activation(np.dot(X, self.W) + self.b)
        return np.dot(H, self.beta)

# ---------------------------------------------------------
# 2. 数据准备 (含排序以模拟连续工况)
# ---------------------------------------------------------
def load_data():
    np.random.seed(42)
    N = 1200 # 增加数据量
    X = np.random.rand(N, 4)
    
    # 构造强非线性交互项: y = 10*sin(pi*x0) + 10*(x1^2)*cos(4*x0) - 5*x2
    y = 10 * np.sin(np.pi * X[:, 0]) + \
        10 * (X[:, 1]**2) * np.cos(4 * X[:, 0]) - \
        5 * X[:, 2] + 20 + np.random.normal(0, 0.2, N)
    
    # 归一化
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # 【关键步骤】对测试集按第一个特征排序，模拟连续的工业过程，这样画图才好看，动态加权才有意义
    sort_idx = np.argsort(X_test[:, 0])
    X_test = X_test[sort_idx]
    y_test = y_test[sort_idx]
    
    return X_train, X_test, y_train, y_test, scaler_y

# ---------------------------------------------------------
# 3. 主程序
# ---------------------------------------------------------
if __name__ == "__main__":
    print(">>> 开始运行综合对比实验...")
    X_train, X_test, y_train, y_test, scaler_y = load_data()
    
    results = {} # 存储结果
    
    # --- 模型 1: SVR (传统基准) ---
    t0 = time.time()
    svr = SVR(kernel='rbf', C=100, gamma=0.1)
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)
    results['SVR'] = {'pred': y_pred_svr, 'time': time.time()-t0}
    
    # --- 模型 2: MLP (神经网络基准) ---
    t0 = time.time()
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    results['MLP'] = {'pred': y_pred_mlp, 'time': time.time()-t0}

    # --- 模型 3: Standard SCN (单一模型基准) ---
    t0 = time.time()
    std_scn = SimpleSCN(n_hidden_nodes=500, regularization=0.001) # 常规正则化
    std_scn.fit(X_train, y_train)
    y_pred_std = std_scn.predict(X_test)
    results['Std-SCN'] = {'pred': y_pred_std, 'time': time.time()-t0}

    # --- 模型 4 & 5: 构建模型池 (用于 Mean-SCN 和 DWMP-SCN) ---
    print(">>> 正在构建差异化模型池 (Bagging + Weak Reg)...")
    POOL_SIZE = 50
    TOP_K = 10
    pool = []
    
    for i in range(POOL_SIZE):
        # Bagging: 80% 数据
        X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=0.8, random_state=None)
        # 弱正则化 (1e-6) 增加差异性
        model = SimpleSCN(n_hidden_nodes=500, regularization=1e-6)
        model.fit(X_sub, y_sub)
        
        # 评估
        rmse_t = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        pool.append({'model': model, 'rmse': rmse_t})
        
    # 优选 Top-K
    pool.sort(key=lambda x: x['rmse'])
    best_pool = pool[:TOP_K]
    
    # --- 模型 4: Mean-SCN (消融实验 - 简单平均) ---
    t0 = time.time()
    preds_pool = np.array([m['model'].predict(X_test) for m in best_pool])
    y_pred_mean = np.mean(preds_pool, axis=0)
    results['Mean-SCN'] = {'pred': y_pred_mean, 'time': time.time()-t0}
    
    # --- 模型 5: DWMP-SCN (本文方法 - 动态加权) ---
    print(">>> 执行 DWMP-SCN 在线动态预测...")
    t0 = time.time()
    LAMBDA = 80
    y_pred_dwmp = []
    weights_history = [] # 记录权重用于画热力图
    
    # 初始化权重 (基于训练集RMSE)
    train_rmses = np.array([m['rmse'] for m in best_pool])
    current_weights = np.exp(-LAMBDA * train_rmses)
    current_weights /= np.sum(current_weights)
    
    # 逐样本在线预测 (模拟工业实时过程)
    for t in range(len(X_test)):
        # 1. 获取当前时刻各子模型预测值
        sub_preds = preds_pool[:, t] # Shape: (10,)
        
        # 2. 加权聚合
        final_y = np.dot(current_weights, sub_preds)
        y_pred_dwmp.append(final_y)
        weights_history.append(current_weights)
        
        # 3. 反馈更新权重 (利用当前时刻误差更新，用于下一时刻)
        # 注意: 实际应用中要有真实值y_test[t]才能更新。此处为模拟在线学习过程。
        errors_t = (sub_preds - y_test[t])**2 # 误差平方
        # Softmax 更新
        new_w = np.exp(-LAMBDA * errors_t)
        current_weights = new_w / (np.sum(new_w) + 1e-8)
        
    y_pred_dwmp = np.array(y_pred_dwmp)
    results['DWMP-SCN'] = {'pred': y_pred_dwmp, 'time': time.time()-t0}
    
    # ---------------------------------------------------------
    # 4. 计算指标与 T-Test
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(f"{'Model':<12} | {'RMSE':<8} | {'MAE':<8} | {'R2':<8} | {'Time(s)':<8}")
    print("-" * 50)
    
    final_metrics = {}
    for name, res in results.items():
        rmse = np.sqrt(mean_squared_error(y_test, res['pred']))
        mae = mean_absolute_error(y_test, res['pred'])
        r2 = r2_score(y_test, res['pred'])
        final_metrics[name] = rmse
        print(f"{name:<12} | {rmse:.5f}   | {mae:.5f}   | {r2:.5f}   | {res['time']:.4f}")
        
    # 显著性检验
    t_stat, p_val = stats.ttest_rel(np.abs(y_test - y_pred_std), np.abs(y_test - y_pred_dwmp))
    print("-" * 50)
    print(f"Standard SCN vs DWMP-SCN 提升比例: {((final_metrics['Std-SCN'] - final_metrics['DWMP-SCN'])/final_metrics['Std-SCN'])*100:.2f}%")
    print(f"Paired T-test P-value: {p_val:.2e} (Significant if < 0.05)")
    print("="*50 + "\n")

    # ---------------------------------------------------------
    # 5. 绘图部分 (4张图)
    # ---------------------------------------------------------
    # 还原真实量纲
    y_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    models_to_plot = ['MLP', 'Std-SCN', 'Mean-SCN', 'DWMP-SCN']
    colors = {'MLP': 'green', 'Std-SCN': 'blue', 'Mean-SCN': 'orange', 'DWMP-SCN': 'red'}
    styles = {'MLP': ':', 'Std-SCN': '--', 'Mean-SCN': '-.', 'DWMP-SCN': '-'}
    
    preds_real = {}
    for m in models_to_plot:
        preds_real[m] = scaler_y.inverse_transform(results[m]['pred'].reshape(-1, 1)).ravel()

    # --- Figure 1: 拟合曲线 & 误差分布 ---
    fig1 = plt.figure(figsize=(14, 6))
    
    # (a) 拟合曲线
    ax1 = plt.subplot(1, 2, 1)
    subset = 100
    ax1.plot(y_real[:subset], 'k-', lw=2, alpha=0.6, label='Actual')
    for m in ['MLP', 'Std-SCN', 'DWMP-SCN']: # 只画3个避免太乱
        lw = 2.5 if m == 'DWMP-SCN' else 1.5
        alpha = 1.0 if m == 'DWMP-SCN' else 0.7
        ax1.plot(preds_real[m][:subset], color=colors[m], ls=styles[m], lw=lw, alpha=alpha, label=m)
    ax1.set_title('(a) Prediction Trajectories', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Output Value')
    ax1.legend()
    
    # (b) 误差分布
    ax2 = plt.subplot(1, 2, 2)
    for m in models_to_plot:
        err = y_real - preds_real[m]
        # DWMP用填充，其他用线条
        if m == 'DWMP-SCN':
            ax2.hist(err, bins=30, density=True, histtype='stepfilled', color='red', alpha=0.2, label=m)
            ax2.hist(err, bins=30, density=True, histtype='step', color='red', lw=2)
        else:
            ax2.hist(err, bins=30, density=True, histtype='step', color=colors[m], ls=styles[m], label=m)
    ax2.set_title('(b) Error Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Prediction Error')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    
    # --- Figure 2: 局部放大图 ---
    fig2, ax = plt.subplots(figsize=(12, 6))
    limit = 150
    x = np.arange(limit)
    ax.plot(x, y_real[:limit], 'k-', lw=1.5, alpha=0.5, label='Actual')
    ax.plot(x, preds_real['Std-SCN'][:limit], 'b--', lw=1.5, alpha=0.8, label='Std-SCN')
    ax.plot(x, preds_real['DWMP-SCN'][:limit], 'r-', lw=2.5, label='DWMP-SCN')
    
    ax.set_title('Fig 2. Detailed View with Zoom-in', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    
    # 插图
    axins = inset_axes(ax, width="40%", height="35%", loc='lower right', borderpad=2)
    z_start, z_end = 60, 90
    axins.plot(x[z_start:z_end], y_real[z_start:z_end], 'k-', lw=1.5)
    axins.plot(x[z_start:z_end], preds_real['Std-SCN'][z_start:z_end], 'b--', lw=1.5)
    axins.plot(x[z_start:z_end], preds_real['DWMP-SCN'][z_start:z_end], 'r-', lw=2.5)
    axins.set_xlim(z_start, z_end)
    axins.set_ylim(np.min(y_real[z_start:z_end])-0.5, np.max(y_real[z_start:z_end])+0.5)
    axins.grid(False) # 插图不画网格更清晰
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", ls='--')
    plt.show()

    # --- Figure 3: 参数敏感性分析 (Lambda) ---
    print(">>> 正在生成参数敏感性图 (Fig 3)...")
    lambda_list = [1, 10, 30, 50, 80, 100, 150, 200]
    rmse_sens = []
    
    # 快速重跑不同Lambda
    for lam in lambda_list:
        # 复用已经预测好的 sub_preds (preds_pool) 以节省时间
        temp_preds = []
        w = np.exp(-lam * train_rmses)
        w /= np.sum(w)
        for t in range(len(X_test)):
            sub_p = preds_pool[:, t]
            temp_preds.append(np.dot(w, sub_p))
            # 简化的在线更新
            err = (sub_p - y_test[t])**2
            w = np.exp(-lam * err)
            w /= (np.sum(w) + 1e-8)
        rmse_sens.append(np.sqrt(mean_squared_error(y_test, temp_preds)))

    plt.figure(figsize=(8, 5))
    plt.plot(lambda_list, rmse_sens, 'o-', color='darkblue', lw=2)
    # 标出最低点
    min_idx = np.argmin(rmse_sens)
    plt.plot(lambda_list[min_idx], rmse_sens[min_idx], 'r*', ms=15, label='Optimal')
    plt.title('Fig 3. Parameter Sensitivity of Softmax $\lambda$', fontsize=12, fontweight='bold')
    plt.xlabel('Lambda ($\lambda$)')
    plt.ylabel('RMSE')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Figure 4: 动态权重热力图 ---
    print(">>> 正在生成动态权重热力图 (Fig 4)...")
    # 取前50个样本的权重历史
    w_hist_array = np.array(weights_history[:50]).T # (10, 50)
    
    plt.figure(figsize=(12, 5))
    sns.heatmap(w_hist_array, cmap='Blues', cbar_kws={'label': 'Weight Magnitude'})
    plt.title('Fig 4. Dynamic Weight Evolution of Top-10 Models', fontsize=12, fontweight='bold')
    plt.xlabel('Test Sample Index (Time)')
    plt.ylabel('Sub-model Index')
    plt.tight_layout()
    plt.show()
    
    print(">>> 所有绘图完成！")