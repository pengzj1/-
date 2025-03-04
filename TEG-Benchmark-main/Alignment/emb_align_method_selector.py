import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    normalized_mutual_info_score,
    adjusted_rand_index
)
from scipy.spatial.distance import cosine
import umap
import matplotlib.pyplot as plt
import seaborn as sns

class AlignmentMethodEvaluator:
    def __init__(self, text_embeddings, graph_embeddings, labels=None):
        """
        初始化对齐方法评估器
        
        Args:
        - text_embeddings (torch.Tensor): 文本嵌入
        - graph_embeddings (torch.Tensor): 图嵌入
        - labels (np.array, optional): ground truth标签，用于聚类评估
        """
        self.text_embeddings = text_embeddings
        self.graph_embeddings = graph_embeddings
        self.labels = labels
        
        # 对齐方法列表
        self.alignment_methods = {
            'predictive_alignment': self._predictive_alignment,
            'latent_space_alignment': self._latent_space_alignment,
            'linear_transformation': self._linear_transformation,
            'canonical_correlation': self._canonical_correlation
        }
    
    def _predictive_alignment(self):
        """
        预测对齐方法
        
        Returns:
        - aligned_embeddings (torch.Tensor)
        """
        from embedding_alignment import PredictiveAlignmentModel
        
        model = PredictiveAlignmentModel(
            self.text_embeddings.shape[1], 
            self.graph_embeddings.shape[1]
        )
        
        optimizer = torch.optim.Adam(model.parameters())
        
        for _ in range(100):
            optimizer.zero_grad()
            pred_text, pred_graph = model(self.text_embeddings, self.graph_embeddings)
            loss = F.mse_loss(pred_text, self.graph_embeddings) + \
                   F.mse_loss(pred_graph, self.text_embeddings)
            loss.backward()
            optimizer.step()
        
        return model.text_to_graph(self.text_embeddings)
    
    def _latent_space_alignment(self):
        """
        潜在空间对齐方法
        
        Returns:
        - aligned_embeddings (torch.Tensor)
        """
        from embedding_alignment import LatentSpaceAligner
        
        aligner = LatentSpaceAligner(
            self.text_embeddings.shape[1], 
            self.graph_embeddings.shape[1]
        )
        
        return aligner(self.text_embeddings, self.graph_embeddings)
    
    def _linear_transformation(self):
        """
        线性变换对齐方法
        
        Returns:
        - aligned_embeddings (torch.Tensor)
        """
        # 使用最小二乘法求解线性变换矩阵
        X = self.text_embeddings.numpy()
        Y = self.graph_embeddings.numpy()
        
        # 求解线性变换矩阵 W，使得 XW ≈ Y
        W = np.linalg.lstsq(X, Y, rcond=None)[0]
        
        return torch.from_numpy(X @ W)
    
    def _canonical_correlation(self):
        """
        典型相关分析（CCA）对齐方法
        
        Returns:
        - aligned_embeddings (torch.Tensor)
        """
        from sklearn.cross_decomposition import CCA
        
        cca = CCA(n_components=min(self.text_embeddings.shape[1], self.graph_embeddings.shape[1]))
        X_c, Y_c = cca.fit_transform(
            self.text_embeddings.numpy(), 
            self.graph_embeddings.numpy()
        )
        
        return torch.from_numpy(X_c)
    
    def evaluate_alignment(self, aligned_embeddings):
        """
        评估对齐质量的多个指标
        
        Args:
        - aligned_embeddings (torch.Tensor): 对齐后的嵌入
        
        Returns:
        - evaluation_metrics (dict): 各种评估指标
        """
        metrics = {}
        
        # 重构误差
        metrics['mse'] = mean_squared_error(
            aligned_embeddings, 
            self.graph_embeddings
        )
        metrics['mae'] = mean_absolute_error(
            aligned_embeddings, 
            self.graph_embeddings
        )
        
        # 余弦相似度
        cosine_distances = [
            cosine(a, b) for a, b in zip(
                aligned_embeddings, 
                self.graph_embeddings
            )
        ]
        metrics['avg_cosine_distance'] = np.mean(cosine_distances)
        
        # 聚类评估（如果有标签）
        if self.labels is not None:
            # 使用UMAP降维进行可视化和聚类
            reducer = umap.UMAP(n_components=2)
            reduced_embeddings = reducer.fit_transform(aligned_embeddings)
            
            # K-means聚类
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=len(np.unique(self.labels)))
            cluster_labels = kmeans.fit_predict(reduced_embeddings)
            
            # 聚类评估指标
            metrics['nmi'] = normalized_mutual_info_score(
                self.labels, cluster_labels
            )
            metrics['ari'] = adjusted_rand_index(
                self.labels, cluster_labels
            )
            
            # 可视化
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(
                reduced_embeddings[:, 0], 
                reduced_embeddings[:, 1], 
                c=cluster_labels, 
                cmap='viridis'
            )
            plt.colorbar(scatter)
            plt.title('Aligned Embeddings Visualization')
            plt.savefig('alignment_visualization.png')
            plt.close()
        
        return metrics
    
    def select_best_alignment_method(self):
        """
        选择最佳对齐方法
        
        Returns:
        - best_method (str): 最佳对齐方法名称
        - evaluation_results (dict): 各方法的评估结果
        """
        evaluation_results = {}
        
        for method_name, method_func in self.alignment_methods.items():
            # 执行对齐
            aligned_embeddings = method_func()
            
            # 评估对齐效果
            metrics = self.evaluate_alignment(aligned_embeddings)
            evaluation_results[method_name] = metrics
            
            print(f"Method: {method_name}")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
        
        # 根据多个指标选择最佳方法
        # 这里以最小重构误差和最大聚类一致性为例
        best_method = min(
            evaluation_results, 
            key=lambda k: (
                evaluation_results[k]['mse'] 
                if 'nmi' not in evaluation_results[k] 
                else -evaluation_results[k]['nmi']
            )
        )
        
        return best_method, evaluation_results

def main():
    # 示例数据生成
    text_emb_dim = 768
    graph_emb_dim = 256
    n_samples = 1000
    
    # 模拟文本和图嵌入
    text_embeddings = torch.randn(n_samples, text_emb_dim)
    graph_embeddings = torch.randn(n_samples, graph_emb_dim)
    
    # 模拟标签（可选）
    from sklearn.datasets import make_blobs
    _, labels = make_blobs(n_samples=n_samples, centers=5)
    
    # 初始化评估器
    evaluator = AlignmentMethodEvaluator(
        text_embeddings, 
        graph_embeddings, 
        labels
    )
    
    # 选择最佳对齐方法
    best_method, results = evaluator.select_best_alignment_method()
    
    print(f"\n最佳对齐方法: {best_method}")
    print("详细评估结果:")
    for method, metrics in results.items():
        print(f"\n{method}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()
# 这个脚本提供了一个全面的对齐方法选择框架，主要特点：

# 对齐方法：

# 预测对齐（Predictive Alignment）
# 潜在空间对齐（Latent Space Alignment）
# 线性变换（Linear Transformation）
# 典型相关分析（Canonical Correlation Analysis）


# 评估指标：

# 均方误差（MSE）
# 平均绝对误差（MAE）
# 平均余弦距离
# 标准化互信息（NMI）
# 调整兰德指数（ARI）


# 可视化：

# 使用UMAP降维
# 聚类可视化
# 保存可视化图像



# 选择最佳方法的逻辑：

# 计算多个评估指标
# 综合考虑重构误差和聚类一致性
# 可根据具体数据集调整权重

# 使用建议：

# 根据数据集特征调整评估指标权重
# 添加更多对齐方法
# 考虑计算效率和可解释性

# 注意事项：

# 需要安装额外库：umap-learn, scikit-learn, matplotlib, seaborn
# 示例中使用随机生成的嵌入，实际使用时替换为真实数据
