import numpy as np
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../2019Happy.csv')
all_features = ['GDP per capita', 'Social support', 'Healthy life expectancy', 
                'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
X_raw = df[all_features].values
y_raw = df['Score'].values.reshape(-1, 1)
select_idx = np.random.permutation(len(X_raw))
n_train = int(0.8 * len(X_raw))
X_train_raw, X_test_raw = X_raw[select_idx[:n_train]], X_raw[select_idx[n_train:]]
y_train, y_test = y_raw[select_idx[:n_train]], y_raw[select_idx[n_train:]]

print(f"训练集大小: {len(y_train)}")
print(f"测试集大小: {len(y_test)}")

def calc_mu_sigma(X, degree):
    m, n = X.shape
    features = []
    for d in range(1, degree + 1):
        for c in combinations_with_replacement(range(n), d):
            tmp = np.prod(X[:, c], axis=1).reshape(-1, 1)
            features.append(tmp)
    X_features = np.hstack(features)
    mu = np.mean(X_features, axis=0)
    sigma = np.std(X_features, axis=0)
    return mu, sigma

def gen_features(X, degree, mu, sigma):
    m, n = X.shape
    features = []
    for d in range(1, degree + 1):
        for c in combinations_with_replacement(range(n), d):
            tmp = np.prod(X[:, c], axis=1).reshape(-1, 1)
            features.append(tmp)
    X_feat = np.hstack(features)
    X_feat_std = (X_feat - mu) / sigma
    bias = np.ones((m, 1))
    return np.hstack([bias, X_feat_std])

def MSE(y, y_pred):
    y, y_pred = y.flatten(), y_pred.flatten()
    diff = y_pred - y
    return np.dot(diff, diff) / len(y)

def loss(y, y_pred, lamda, weight):
    y, y_pred = y.flatten(), y_pred.flatten()
    mse = np.dot(y_pred - y, y_pred - y) / (2 * len(y))
    l2 = (lamda / 2) * np.dot(weight[1:], weight[1:])
    return mse + l2

def train(X_train, y_train, X_test, y_test, degree, lr, epochs, lamda):
    mu, sigma = calc_mu_sigma(X_train, degree)
    X_train_feat = gen_features(X_train, degree, mu, sigma)
    X_test_feat = gen_features(X_test, degree, mu, sigma)

    n, m = X_train_feat.shape
    W = np.zeros(m)
    train_loss_hist, test_mse_hist = [], []
    y_tr_flat, y_te_flat = y_train.flatten(), y_test.flatten()

    for i in range(epochs):
        y_pred = X_train_feat @ W
        curr_loss = loss(y_tr_flat, y_pred, lamda, W)
        train_loss_hist.append(curr_loss)
        
        grad = (X_train_feat.T @ (y_pred - y_tr_flat)) / n
        W_reg = np.copy(W)
        W_reg[0] = 0
        grad += lamda * W_reg
        W -= lr * grad
        
        if i % 10 == 0:
            te_pred = X_test_feat @ W
            mse_val = MSE(y_te_flat, te_pred)
            test_mse_hist.append((i, mse_val))
            
    return W, mu, sigma, train_loss_hist, test_mse_hist

degree = 2
W, mu, sigma, train_loss, test_mse = train(X_train_raw, y_train, X_test_raw, y_test, 
                                          degree=degree, lr=0.01, epochs=500, lamda=0.1)

iters, mses = zip(*test_mse)

fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Train Loss (L2 Reg)', color=color1, fontsize=12)
ax1.plot(range(len(train_loss)), train_loss, color=color1, linewidth=2, label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.6)

# --- 绘制第二个轴：Test MSE ---
ax2 = ax1.twinx()  # 实例化一个共享相同 x 轴的第二个轴
color2 = 'tab:red'
ax2.set_ylabel('Test MSE', color=color2, fontsize=12)
ax2.plot(iters, mses, color=color2, marker='o', markersize=4, linestyle='-', label='Test MSE')
ax2.tick_params(axis='y', labelcolor=color2)

# 设置标题
plt.title('Training Progress: Loss vs. Test MSE (Dual Axis)', fontsize=14)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

fig.tight_layout()
plt.show()
