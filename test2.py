import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import accuracy_score, silhouette_score
# from scipy.stats import entropy
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 讀取資料
data = pd.read_csv('sizes3 (with class label).csv')
# 取出 features
X = data.iloc[:, :-1]
# 取出 class label
y = data.iloc[:, -1]
# 使用 MinMaxScaler 進行縮放
minmax_scaler = MinMaxScaler()
X_minmax_scaled = minmax_scaler.fit_transform(X)
X = X_minmax_scaled
print(X)

# 繪製散布圖
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=25, color='blue', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=25, color='green', label='Class 1')
plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], s=25, color='yellow', label='Class 2')
plt.scatter(X[y == 3][:, 0], X[y == 3][:, 1], s=25, color='red', label='Class 3')
plt.title('Original Data Scatter Plot')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
# K-means演算法
start_time = time.time()
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)
kmeans_time = time.time() - start_time
kmeans_sse = kmeans.inertia_

# 階層式分群演算法
start_time = time.time()
# linkage method 和 metric 可以根據實際需求調整
linkage_matrix = linkage(X, method='average', metric='euclidean')
hierarchical_labels = fcluster(linkage_matrix, t=4, criterion='maxclust') - 1  # t 表示分成的群數
hierarchical_time = time.time() - start_time
hierarchical_sse = 0
for cluster_id in range(4):
    cluster_points = X[hierarchical_labels == cluster_id]
    centroid = cluster_points.mean(axis=0)
    hierarchical_sse += ((cluster_points - centroid) ** 2).sum()
# 將 hierarchical_sse 轉換為數值
if isinstance(hierarchical_sse, pd.Series):
    hierarchical_sse_value = hierarchical_sse.mean()  # 或者可以使用其他統計量
else:
    hierarchical_sse_value = hierarchical_sse

# DBSCAN1
start_time = time.time()
dbscan = DBSCAN(eps=0.1, min_samples=5)
dbscan1_labels = dbscan.fit_predict(X)
dbscan1_time = time.time() - start_time

dbscan1_sse = 0
for label in set(dbscan1_labels):
    if label == -1:  # 跳過噪音點
        continue
    cluster_points = X[dbscan1_labels == label]
    centroid = cluster_points.mean(axis=0)
    dbscan1_sse += ((cluster_points - centroid) ** 2).sum()

# 將 dbscan1_sse 轉換為數值
if isinstance(dbscan1_sse, pd.Series):
    dbscan1_sse_value = dbscan1_sse.mean()  # 或者可以使用其他統計量
else:
    dbscan1_sse_value = dbscan1_sse

# 計算密度
# dbscan1_density = len(X) / len(set(dbscan1_labels))

# 計算噪音點比例
# dbscan1_noise_ratio = list(dbscan1_labels).count(-1) / len(dbscan1_labels)

# DBSCAN2
start_time = time.time()
dbscan = DBSCAN(eps=0.15, min_samples=5)
dbscan2_labels = dbscan.fit_predict(X)
dbscan2_time = time.time() - start_time

dbscan2_sse = 0
for label in set(dbscan2_labels):
    if label == -1:  # 跳過噪音點
        continue
    cluster_points = X[dbscan2_labels == label]
    centroid = cluster_points.mean(axis=0)
    dbscan2_sse += ((cluster_points - centroid) ** 2).sum()

# 將 dbscan2_sse 轉換為數值
if isinstance(dbscan2_sse, pd.Series):
    dbscan2_sse_value = dbscan2_sse.mean()  # 或者可以使用其他統計量
else:
    dbscan2_sse_value = dbscan2_sse
# 計算密度
# dbscan2_density = len(X) / len(set(dbscan2_labels))

# 計算噪音點比例
# dbscan2_noise_ratio = list(dbscan2_labels).count(-1) / len(dbscan2_labels)


# 計算指標
def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    predicted_entropy = 0
    for label in set(predicted_labels):
        p = (predicted_labels == label).sum() / len(predicted_labels)
        predicted_entropy -= p * np.log2(p)
    return accuracy, predicted_entropy

kmeans_accuracy, kmeans_predicted_entropy = calculate_metrics(y, kmeans_labels)
hierarchical_accuracy, hierarchical_predicted_entropy = calculate_metrics(y, hierarchical_labels)
dbscan1_accuracy, dbscan1_predicted_entropy = calculate_metrics(y, dbscan1_labels)
dbscan2_accuracy, dbscan2_predicted_entropy = calculate_metrics(y, dbscan2_labels)

# 輸出結果
print("K-means:")
print(f"Time: {kmeans_time:.4f} seconds")
print(f"Accuracy: {kmeans_accuracy:.4f}")
print(f"Predicted_Entropy: {kmeans_predicted_entropy:.4f}")
print(f"SSE: {kmeans_sse:.4f}")

print("\nHierarchical:")
print(f"Time: {hierarchical_time:.4f} seconds")
print(f"Accuracy: {hierarchical_accuracy:.4f}")
print(f"Predicted_Entropy: {hierarchical_predicted_entropy:.4f}")
print(f"SSE: {hierarchical_sse_value:.4f}")

print("\nDBSCAN1:")
print(f"Time: {dbscan1_time:.4f} seconds")
print(f"Accuracy: {dbscan1_accuracy:.4f}")
print(f"Predicted_Entropy: {dbscan1_predicted_entropy:.4f}")
# print(f"Density: {dbscan1_density:.4f}")
# print(f"Noise Ratio: {dbscan1_noise_ratio:.4f}")
print(f"SSE: {dbscan1_sse_value:.4f}")

print("\nDBSCAN2:")
print(f"Time: {dbscan2_time:.4f} seconds")
print(f"Accuracy: {dbscan2_accuracy:.4f}")
print(f"Predicted_Entropy: {dbscan2_predicted_entropy:.4f}")
# print(f"Density: {dbscan2_density:.4f}")
# print(f"Noise Ratio: {dbscan2_noise_ratio:.4f}")
print(f"SSE: {dbscan2_sse_value:.4f}")

# 繪製分群結果圖
def plot_clusters(data, labels, title):
    plt.figure(figsize=(8, 6))
    markers = {0: '1', 1: '2', 2: '3', 3: '4'}

    for i, label in enumerate(set(labels)):
        cluster_points = data[labels == label]
        if i in markers:
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                label=f'Cluster {markers[i]}',
                marker=markers[i],  # 使用標記 '1', '2', '3', '4'
                s=25
            )

    plt.title(title)
    plt.legend(markers.values())  # 修改這一行
    plt.show()


plot_clusters(X, kmeans_labels, 'K-means Clustering')
plot_clusters(X, hierarchical_labels, 'Hierarchical Clustering')
plot_clusters(X, dbscan1_labels, 'DBSCAN1 Clustering')
plot_clusters(X, dbscan2_labels, 'DBSCAN2 Clustering')


