import torch
import torch.nn as nn

class EnhancedMultiModalAligner(nn.Module):
    """增强型多模态对齐模型：整合预测对齐与潜在空间对齐，添加注意力机制"""
    
    def __init__(self, graph_dim, text_dim, latent_dim=256, num_classes=None, dropout=0.2):
        """
        参数:
            graph_dim: 图嵌入维度
            text_dim: 文本嵌入维度
            latent_dim: 潜在空间维度
            num_classes: 分类任务的类别数量
            dropout: Dropout比例
        """
        super(EnhancedMultiModalAligner, self).__init__()
        
        # 投影层：将不同模态映射到同一潜在空间
        self.graph_projector = nn.Sequential(
            nn.Linear(graph_dim, latent_dim*2),
            nn.LayerNorm(latent_dim*2),
            nn.GELU(),  # 使用GELU替代ReLU获得更平滑的梯度
            nn.Dropout(dropout),
            nn.Linear(latent_dim*2, latent_dim)
        )
        
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, latent_dim*2),
            nn.LayerNorm(latent_dim*2),  # 使用LayerNorm增强训练稳定性
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(latent_dim*2, latent_dim)
        )
        
        # 跨模态注意力机制
        self.graph_to_text_attention = nn.MultiheadAttention(
            latent_dim, 4, dropout=dropout, batch_first=True
        )
        self.text_to_graph_attention = nn.MultiheadAttention(
            latent_dim, 4, dropout=dropout, batch_first=True
        )
        
        # 融合层
        self.graph_fusion = nn.Sequential(
            nn.Linear(latent_dim*2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout)
        )
        
        self.text_fusion = nn.Sequential(
            nn.Linear(latent_dim*2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout)
        )
        
        # 预测头：用于预测任务（如分类）
        if num_classes:
            self.graph_predictor = nn.Linear(latent_dim, num_classes)
            self.text_predictor = nn.Linear(latent_dim, num_classes)
            # 融合预测头
            self.fusion_predictor = nn.Linear(latent_dim*2, num_classes)
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.temperature = 0.07  # 对比学习温度参数
    
    def forward(self, graph_embeds, text_embeds):
        """前向传播，添加跨模态注意力和融合"""
        batch_size = graph_embeds.shape[0]
        
        # 投影到潜在空间
        graph_latent = self.graph_projector(graph_embeds)
        text_latent = self.text_projector(text_embeds)
        
        # 应用跨模态注意力
        graph_attended, _ = self.graph_to_text_attention(
            graph_latent, text_latent, text_latent
        )
        text_attended, _ = self.text_to_graph_attention(
            text_latent, graph_latent, graph_latent
        )
        
        # 融合表示
        graph_fused = self.graph_fusion(torch.cat([graph_latent, graph_attended], dim=1))
        text_fused = self.text_fusion(torch.cat([text_latent, text_attended], dim=1))
        
        # L2归一化嵌入（对对比学习很重要）
        graph_latent_norm = nn.functional.normalize(graph_fused, p=2, dim=1)
        text_latent_norm = nn.functional.normalize(text_fused, p=2, dim=1)
        
        results = {
            "graph_latent": graph_fused,
            "text_latent": text_fused,
            "graph_latent_norm": graph_latent_norm,
            "text_latent_norm": text_latent_norm,
            "joint_latent": torch.cat([graph_fused, text_fused], dim=1)
        }
        
        # 如果有分类任务，计算预测结果
        if self.num_classes:
            results["graph_preds"] = self.graph_predictor(graph_fused)
            results["text_preds"] = self.text_predictor(text_fused)
            # 融合预测
            results["fusion_preds"] = self.fusion_predictor(results["joint_latent"])
        
        return results
    
    def compute_infoNCE_loss(self, graph_latent, text_latent, labels=None, hard_negative_factor=0.1):
        """使用InfoNCE损失进行对比学习"""
        batch_size = graph_latent.shape[0]
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(graph_latent, text_latent.transpose(0, 1)) / self.temperature
        
        # InfoNCE损失
        eye = torch.eye(batch_size, device=sim_matrix.device)
        
        # 如果有标签信息，构造加权的正负样本对
        if labels is not None:
            pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            # 去掉对角线上的自身匹配
            pos_mask = pos_mask * (1 - eye)
            
            # 困难负样本加权
            neg_mask = 1 - pos_mask - eye
            
            # 计算困难负样本（相似度高但不是正样本的）
            hard_negatives = sim_matrix * neg_mask
            # 选择最困难的负样本
            hardest_negatives = torch.topk(hard_negatives, k=min(3, batch_size-2), dim=1)[0]
            
            # 正样本损失
            pos_per_row = pos_mask.sum(1)
            pos_loss = torch.zeros(batch_size, device=sim_matrix.device)
            for i in range(batch_size):
                if pos_per_row[i] > 0:  # 确保有正样本
                    pos_loss[i] = -torch.log(
                        torch.exp(sim_matrix[i, pos_mask[i] > 0]).sum() / 
                        (torch.exp(sim_matrix[i]).sum())
                    )
            
            # 困难负样本损失
            hard_neg_loss = -torch.log(1 - torch.sigmoid(hardest_negatives)).mean()
            
            # 总损失
            loss = pos_loss.mean() + hard_negative_factor * hard_neg_loss
        else:
            # 标准InfoNCE损失
            labels = torch.arange(batch_size, device=sim_matrix.device)
            loss_i = nn.CrossEntropyLoss()(sim_matrix, labels)
            loss_t = nn.CrossEntropyLoss()(sim_matrix.t(), labels)
            loss = (loss_i + loss_t) / 2
            
        return loss
    
    def compute_prediction_loss(self, graph_preds, text_preds, fusion_preds, labels, 
                                consistency_weight=0.3, fusion_weight=0.5):
        """计算预测对齐损失，包括融合预测"""
        # 任务损失（如分类交叉熵）
        graph_task_loss = nn.CrossEntropyLoss()(graph_preds, labels)
        text_task_loss = nn.CrossEntropyLoss()(text_preds, labels)
        fusion_task_loss = nn.CrossEntropyLoss()(fusion_preds, labels)
        
        # 预测一致性损失（KL散度，更适合概率分布对齐）
        graph_probs = nn.functional.softmax(graph_preds, dim=1)
        text_probs = nn.functional.softmax(text_preds, dim=1)
        
        kl_g2t = nn.KLDivLoss(reduction="batchmean")(
            nn.functional.log_softmax(text_preds, dim=1), 
            graph_probs
        )
        kl_t2g = nn.KLDivLoss(reduction="batchmean")(
            nn.functional.log_softmax(graph_preds, dim=1), 
            text_probs
        )
        consistency_loss = (kl_g2t + kl_t2g) / 2
        
        # 总预测损失
        pred_loss = graph_task_loss + text_task_loss + fusion_weight * fusion_task_loss + \
                    consistency_weight * consistency_loss
        
        return pred_loss, {
            "graph_task_loss": graph_task_loss.item(),
            "text_task_loss": text_task_loss.item(),
            "fusion_task_loss": fusion_task_loss.item(),
            "consistency_loss": consistency_loss.item()
        }
    
    def compute_total_loss(self, batch, pred_weight=1.0, contrastive_weight=1.0):
        """计算总损失：预测对齐 + 潜在空间对齐"""
        graph_embeds, text_embeds, labels = batch
        
        # 前向传播
        outputs = self.forward(graph_embeds, text_embeds)
        
        losses = {}
        total_loss = 0
        
        # 潜在空间对齐损失（InfoNCE对比学习）
        contrastive_loss = self.compute_infoNCE_loss(
            outputs["graph_latent_norm"], 
            outputs["text_latent_norm"],
            labels
        )
        losses["contrastive_loss"] = contrastive_loss.item()
        total_loss += contrastive_weight * contrastive_loss
        
        # 预测对齐损失
        if self.num_classes:
            pred_loss, pred_losses = self.compute_prediction_loss(
                outputs["graph_preds"],
                outputs["text_preds"],
                outputs["fusion_preds"],
                labels
            )
            losses.update(pred_losses)
            total_loss += pred_weight * pred_loss
        
        return total_loss, losses, outputs