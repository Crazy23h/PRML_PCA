"""
基于PCA（特征脸）的人脸识别系统
使用Olivetti人脸数据集，从零开始实现PCA算法
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示窗口
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import accuracy_score

# ============================================================
# 1. 数据准备
# ============================================================

print("正在加载Olivetti人脸数据集...")
faces = fetch_olivetti_faces(shuffle=False)
X = faces.data          # 形状: (400, 4096)，每行是一张展平的64x64人脸图像
y = faces.target        # 形状: (400,)，标签0~39，每人10张

# 手动划分训练集和测试集：每人前7张用于训练，后3张用于测试
n_subjects = 40         # 总人数
n_train_per = 7         # 每人训练图像数
n_test_per = 3          # 每人测试图像数
n_per = 10              # 每人总图像数

train_idx = []
test_idx = []
for i in range(n_subjects):
    start = i * n_per
    train_idx.extend(range(start, start + n_train_per))
    test_idx.extend(range(start + n_train_per, start + n_per))

X_train = X[train_idx]  # 形状: (280, 4096)
y_train = y[train_idx]
X_test = X[test_idx]    # 形状: (120, 4096)
y_test = y[test_idx]

print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# ============================================================
# 2. 从零开始实现PCA
# ============================================================

def compute_pca(X_train, n_components):
    """
    从零实现PCA，使用X^T X技巧或SVD避免计算巨大的协方差矩阵。

    原始协方差矩阵 C = (1/N) * Phi * Phi^T，其中Phi是中心化数据矩阵，
    C的维度为 d×d（4096×4096），直接求本征值非常耗时且占用大量内存。

    X^T X 技巧：
    - 计算小矩阵 C_tilde = Phi^T * Phi，维度为 N×N（280×280）
    - 求解 C_tilde 的特征向量 v_k
    - 原始协方差矩阵的特征向量为 u_k = Phi * v_k（需归一化）
    - 这将计算复杂度从 O(d^3) 降低到 O(N^3)

    参数:
        X_train: 训练数据，形状 (N, d)
        n_components: 保留的主成分数量

    返回:
        mean_face: 平均脸，形状 (d,)
        components: 主成分（特征脸），形状 (n_components, d)
        explained_variance: 对应的特征值
    """
    N, d = X_train.shape

    # 步骤1：计算平均脸
    mean_face = np.mean(X_train, axis=0)  # 形状: (d,)

    # 步骤2：中心化数据（减去均值）
    Phi = X_train - mean_face  # 形状: (N, d)

    # 步骤3：使用SVD分解（等价于X^T X技巧，但数值更稳定）
    # np.linalg.svd 返回 U, S, Vt，其中：
    #   U: (N, N) 左奇异向量（对应样本方向）
    #   S: 奇异值，满足 S^2 / (N-1) = 特征值
    #   Vt: (d, d) 右奇异向量（即主成分方向，我们需要的特征脸）
    # 使用 full_matrices=False 只计算需要的部分，节省内存
    U, S, Vt = np.linalg.svd(Phi, full_matrices=False)

    # 步骤4：提取前 n_components 个主成分
    components = Vt[:n_components]          # 形状: (n_components, d)
    explained_variance = (S[:n_components] ** 2) / (N - 1)

    return mean_face, components, explained_variance


def project(X, mean_face, components):
    """
    将数据投影到PCA子空间（降维）

    参数:
        X: 输入数据，形状 (N, d)
        mean_face: 平均脸，形状 (d,)
        components: 主成分，形状 (k, d)

    返回:
        X_proj: 投影后的低维表示，形状 (N, k)
    """
    Phi = X - mean_face              # 中心化
    X_proj = Phi @ components.T      # 投影：y = U_k^T * phi
    return X_proj


def reconstruct(X_proj, mean_face, components):
    """
    从低维表示重建图像

    参数:
        X_proj: 投影后的低维表示，形状 (N, k)
        mean_face: 平均脸，形状 (d,)
        components: 主成分，形状 (k, d)

    返回:
        X_recon: 重建的图像，形状 (N, d)
    """
    X_recon = X_proj @ components + mean_face   # 重建：x_hat = mean + U_k * y
    return X_recon


def nearest_neighbor_classify(X_train_proj, y_train, X_test_proj):
    """
    基于欧氏距离的最近邻分类器

    参数:
        X_train_proj: 训练集投影，形状 (N_train, k)
        y_train: 训练标签，形状 (N_train,)
        X_test_proj: 测试集投影，形状 (N_test, k)

    返回:
        y_pred: 预测标签，形状 (N_test,)
    """
    y_pred = []
    for test_vec in X_test_proj:
        # 计算测试样本与所有训练样本的欧氏距离
        dists = np.linalg.norm(X_train_proj - test_vec, axis=1)
        # 找到最近邻的标签
        nn_idx = np.argmin(dists)
        y_pred.append(y_train[nn_idx])
    return np.array(y_pred)


# ============================================================
# 3. 可视化平均脸和前10个特征脸
# ============================================================

print("\n计算并可视化特征脸...")
mean_face, components_100, explained_variance_100 = compute_pca(X_train, n_components=100)

fig, axes = plt.subplots(2, 6, figsize=(15, 5))
fig.suptitle('Mean Face and Top 10 Eigenfaces', fontsize=14)

# 显示平均脸
axes[0, 0].imshow(mean_face.reshape(64, 64), cmap='gray')
axes[0, 0].set_title('Mean Face')
axes[0, 0].axis('off')

# 显示前10个特征脸
for i in range(10):
    row = (i + 1) // 6
    col = (i + 1) % 6
    axes[row, col].imshow(components_100[i].reshape(64, 64), cmap='gray')
    axes[row, col].set_title(f'Eigenface {i+1}')
    axes[row, col].axis('off')

# 隐藏多余的子图
axes[1, 5].axis('off')

plt.tight_layout()
plt.savefig('eigenfaces.png', dpi=150, bbox_inches='tight')
plt.close()
print("特征脸图像已保存为 eigenfaces.png")

# ============================================================
# 4. 实验比较：不同k值对应的识别准确率
# ============================================================

print("\n开始实验：测试不同k值的识别准确率...")
k_values = [10, 20, 30, 40, 50, 80, 100]
accuracies = []

for k in k_values:
    # 使用前k个主成分
    mean_face_k, components_k, _ = compute_pca(X_train, n_components=k)

    # 投影训练集和测试集到k维空间
    X_train_proj = project(X_train, mean_face_k, components_k)
    X_test_proj = project(X_test, mean_face_k, components_k)

    # 最近邻分类
    y_pred = nearest_neighbor_classify(X_train_proj, y_train, X_test_proj)

    # 计算准确率
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"  k = {k:3d}: 准确率 = {acc:.4f} ({acc*100:.1f}%)")

# 绘制准确率 vs k 折线图
plt.figure(figsize=(8, 5))
plt.plot(k_values, [a * 100 for a in accuracies], 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Principal Components (k)', fontsize=12)
plt.ylabel('Recognition Accuracy (%)', fontsize=12)
plt.title('Face Recognition Accuracy vs. Number of Principal Components', fontsize=13)
plt.xticks(k_values)
plt.grid(True, alpha=0.3)
for k, acc in zip(k_values, accuracies):
    plt.annotate(f'{acc*100:.1f}%', (k, acc * 100),
                 textcoords="offset points", xytext=(0, 8),
                 ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('accuracy_vs_k.png', dpi=150, bbox_inches='tight')
plt.close()
print("准确率折线图已保存为 accuracy_vs_k.png")

# ============================================================
# 5. 预测示例可视化（成功与失败案例）
# ============================================================

print("\n生成预测示例图...")

# 使用k=50的结果进行可视化
k_vis = 50
mean_face_vis, components_vis, _ = compute_pca(X_train, n_components=k_vis)
X_train_proj_vis = project(X_train, mean_face_vis, components_vis)
X_test_proj_vis = project(X_test, mean_face_vis, components_vis)
y_pred_vis = nearest_neighbor_classify(X_train_proj_vis, y_train, X_test_proj_vis)

# 找出预测正确和错误的样本
correct_idx = np.where(y_pred_vis == y_test)[0]
wrong_idx = np.where(y_pred_vis != y_test)[0]

# 随机选取4个成功和4个失败（若失败不足4个则取所有）
np.random.seed(42)
n_correct_show = min(4, len(correct_idx))
n_wrong_show = min(4, len(wrong_idx))
selected_correct = np.random.choice(correct_idx, n_correct_show, replace=False)
selected_wrong = np.random.choice(wrong_idx, n_wrong_show, replace=False) if len(wrong_idx) > 0 else []

selected_idx = list(selected_correct) + list(selected_wrong)
labels_correct = ['✓'] * n_correct_show + ['✗'] * n_wrong_show

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle(f'Prediction Examples (k={k_vis}): ✓=Correct, ✗=Wrong', fontsize=13)

for i, (idx, mark) in enumerate(zip(selected_idx, labels_correct)):
    row = i // 4
    col = i % 4
    axes[row, col].imshow(X_test[idx].reshape(64, 64), cmap='gray')
    color = 'green' if mark == '✓' else 'red'
    axes[row, col].set_title(
        f'{mark} True: {y_test[idx]}, Pred: {y_pred_vis[idx]}',
        color=color, fontsize=10
    )
    axes[row, col].axis('off')

# 隐藏未使用的子图
for i in range(len(selected_idx), 8):
    axes[i // 4, i % 4].axis('off')

plt.tight_layout()
plt.savefig('prediction_examples.png', dpi=150, bbox_inches='tight')
plt.close()
print("预测示例图已保存为 prediction_examples.png")

# ============================================================
# 6. 不同k值下的人脸重建效果可视化
# ============================================================

print("\n生成不同k值下的重建效果图...")
k_recon_values = [5, 10, 30, 50, 80, 100]

# 选取一个测试样本用于重建展示
sample_idx = 0
sample = X_test[sample_idx:sample_idx+1]  # 形状: (1, 4096)

fig, axes = plt.subplots(1, len(k_recon_values) + 1, figsize=(16, 3))

# 显示原始图像
axes[0].imshow(sample[0].reshape(64, 64), cmap='gray')
axes[0].set_title('Original', fontsize=10)
axes[0].axis('off')

# 显示不同k值下的重建结果
for i, k in enumerate(k_recon_values):
    mean_face_r, components_r, _ = compute_pca(X_train, n_components=k)
    proj = project(sample, mean_face_r, components_r)
    recon = reconstruct(proj, mean_face_r, components_r)
    axes[i + 1].imshow(recon[0].reshape(64, 64), cmap='gray')
    axes[i + 1].set_title(f'k={k}', fontsize=10)
    axes[i + 1].axis('off')

plt.suptitle('Face Reconstruction at Different k Values', fontsize=13)
plt.tight_layout()
plt.savefig('reconstruction.png', dpi=150, bbox_inches='tight')
plt.close()
print("重建效果图已保存为 reconstruction.png")

# ============================================================
# 7. 输出汇总结果
# ============================================================

print("\n===== 实验结果汇总 =====")
print(f"{'k值':>6}  {'准确率':>10}")
print("-" * 20)
for k, acc in zip(k_values, accuracies):
    print(f"{k:>6}  {acc*100:>9.1f}%")

best_k = k_values[np.argmax(accuracies)]
best_acc = max(accuracies)
print(f"\n最优k值: {best_k}，最高准确率: {best_acc*100:.1f}%")
print("\n所有图像已保存：eigenfaces.png, accuracy_vs_k.png, prediction_examples.png, reconstruction.png")
