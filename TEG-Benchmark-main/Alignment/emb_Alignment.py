import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingAligner:
    def __init__(self, text_emb_dim, graph_emb_dim, hidden_dim=128):
        """
        初始化嵌入对齐器
        
        Args:
        - text_emb_dim (int): 文本嵌入维度
        - graph_emb_dim (int): 图嵌入维度
        - hidden_dim (int): 隐藏层维度
        """
        self.predictive_alignment_model = PredictiveAlignmentModel(text_emb_dim, graph_emb_dim, hidden_dim)
        self.latent_space_aligner = LatentSpaceAligner(text_emb_dim, graph_emb_dim)

    def predictive_alignment(self, text_embeddings, graph_embeddings, epochs=100, lr=1e-3):
        """
        预测对齐方法
        
        Args:
        - text_embeddings (torch.Tensor): 文本嵌入
        - graph_embeddings (torch.Tensor): 图嵌入
        - epochs (int): 训练轮数
        - lr (float): 学习率
        
        Returns:
        - aligned_text_embeddings (torch.Tensor)
        - aligned_graph_embeddings (torch.Tensor)
        """
        optimizer = Adam(self.predictive_alignment_model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            # 前向传播
            pred_text_emb, pred_graph_emb = self.predictive_alignment_model(text_embeddings, graph_embeddings)
            
            # 计算损失
            loss = F.mse_loss(pred_text_emb, graph_embeddings) + F.mse_loss(pred_graph_emb, text_embeddings)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
        
        return self.predictive_alignment_model.align(text_embeddings, graph_embeddings)

    def latent_space_alignment(self, text_embeddings, graph_embeddings, epochs=100, lr=1e-3):
        """
        潜在空间对齐方法
        
        Args:
        - text_embeddings (torch.Tensor): 文本嵌入
        - graph_embeddings (torch.Tensor): 图嵌入
        - epochs (int): 训练轮数
        - lr (float): 学习率
        
        Returns:
        - aligned_embeddings (torch.Tensor)
        """
        aligned_embeddings = self.latent_space_aligner(text_embeddings, graph_embeddings, epochs, lr)
        return aligned_embeddings

    def compute_alignment_score(self, text_emb, graph_emb):
        """
        计算文本和图嵌入的对齐分数
        
        Args:
        - text_emb (torch.Tensor): 文本嵌入
        - graph_emb (torch.Tensor): 图嵌入
        
        Returns:
        - alignment_score (float): 对齐分数
        """
        # 使用余弦相似度作为对齐分数
        text_emb_np = text_emb.detach().numpy()
        graph_emb_np = graph_emb.detach().numpy()
        
        similarity_matrix = cosine_similarity(text_emb_np, graph_emb_np)
        return np.mean(similarity_matrix)


class PredictiveAlignmentModel(nn.Module):
    def __init__(self, text_emb_dim, graph_emb_dim, hidden_dim=128):
        super().__init__()
        # 文本嵌入到图嵌入的映射网络
        self.text_to_graph = nn.Sequential(
            nn.Linear(text_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, graph_emb_dim)
        )
        
        # 图嵌入到文本嵌入的映射网络
        self.graph_to_text = nn.Sequential(
            nn.Linear(graph_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, text_emb_dim)
        )
    
    def forward(self, text_embeddings, graph_embeddings):
        # 预测图嵌入
        pred_graph_emb = self.text_to_graph(text_embeddings)
        # 预测文本嵌入
        pred_text_emb = self.graph_to_text(graph_embeddings)
        
        return pred_text_emb, pred_graph_emb
    
    def align(self, text_embeddings, graph_embeddings):
        # 生成对齐后的嵌入
        aligned_text_emb = self.text_to_graph(text_embeddings)
        aligned_graph_emb = self.graph_to_text(graph_embeddings)
        
        return aligned_text_emb, aligned_graph_emb


class LatentSpaceAligner(nn.Module):
    def __init__(self, text_emb_dim, graph_emb_dim):
        super().__init__()
        # 对齐变换矩阵
        self.alignment_matrix = nn.Parameter(torch.randn(text_emb_dim, graph_emb_dim))
    
    def forward(self, text_embeddings, graph_embeddings, epochs=100, lr=1e-3):
        optimizer = Adam([self.alignment_matrix], lr=lr)
        
        for epoch in range(epochs):
            # 对齐变换
            aligned_text_emb = torch.matmul(text_embeddings, self.alignment_matrix)
            
            # 对齐损失 - 最小化对齐文本嵌入和图嵌入之间的距离
            loss = F.mse_loss(aligned_text_emb, graph_embeddings)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Alignment Loss: {loss.item()}")
        
        return torch.matmul(text_embeddings, self.alignment_matrix)


def main():
    # 示例用法
    text_emb_dim = 768  # BERT/LLaMA嵌入维度
    graph_emb_dim = 256  # 图嵌入维度
    
    # 生成随机嵌入作为示例
    text_embeddings = torch.randn(100, text_emb_dim)
    graph_embeddings = torch.randn(100, graph_emb_dim)
    
    # 初始化对齐器
    aligner = EmbeddingAligner(text_emb_dim, graph_emb_dim)
    
    # 预测对齐
    print("Predictive Alignment:")
    aligned_text_pred, aligned_graph_pred = aligner.predictive_alignment(text_embeddings, graph_embeddings)
    
    # 潜在空间对齐
    print("\nLatent Space Alignment:")
    aligned_embeddings = aligner.latent_space_alignment(text_embeddings, graph_embeddings)
    
    # 计算对齐分数
    pred_alignment_score = aligner.compute_alignment_score(aligned_text_pred, graph_embeddings)
    latent_alignment_score = aligner.compute_alignment_score(aligned_embeddings, graph_embeddings)
    
    print(f"\nPredictive Alignment Score: {pred_alignment_score}")
    print(f"Latent Space Alignment Score: {latent_alignment_score}")


if __name__ == "__main__":
    main()
