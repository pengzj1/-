import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

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


class WeightedRandomSampler:
    """加权随机采样器：处理类别不平衡问题"""
    def __init__(self, labels, replacement=True):
        self.labels = labels
        self.replacement = replacement
        self.weights = self._compute_weights(labels)
    
    def _compute_weights(self, labels):
        """计算每个样本的权重（类频率的倒数）"""
        label_counts = torch.bincount(labels)
        class_weights = 1.0 / label_counts.float()
        # 处理可能的零除
        class_weights[torch.isinf(class_weights)] = 0
        
        sample_weights = class_weights[labels]
        return sample_weights
    
    def __call__(self, n_samples):
        """返回加权采样的索引"""
        return torch.multinomial(self.weights, n_samples, self.replacement)


def train_aligner_with_scheduler(model, train_loader, val_loader, num_epochs=50, 
                                 lr=0.001, weight_decay=1e-5, warmup_steps=100):
    """使用学习率调度、梯度裁剪和早停的训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 加入权重衰减正则化
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 温热学习率调度
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # 早停检测
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        all_losses = {}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # 将数据移到设备上
            batch = [item.to(device) for item in batch]
            
            optimizer.zero_grad()
            loss, losses, _ = model.compute_total_loss(batch)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # 更新进度条
            progress_bar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
            
            epoch_train_loss += loss.item()
            
            # 累积各种损失值
            for k, v in losses.items():
                if k not in all_losses:
                    all_losses[k] = []
                all_losses[k].append(v)
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # 打印详细损失信息
        loss_str = " - ".join([f"{k}: {np.mean(v):.4f}" for k, v in all_losses.items()])
        print(f"Epoch {epoch+1} Train - Loss: {avg_train_loss:.4f} - {loss_str}")
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        val_losses = {}
        
        with torch.no_grad():
            for batch in val_loader:
                batch = [item.to(device) for item in batch]
                loss, losses, _ = model.compute_total_loss(batch)
                epoch_val_loss += loss.item()
                
                # 累积验证损失
                for k, v in losses.items():
                    if k not in val_losses:
                        val_losses[k] = []
                    val_losses[k].append(v)
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_loss_str = " - ".join([f"{k}: {np.mean(v):.4f}" for k, v in val_losses.items()])
        print(f"Epoch {epoch+1} Val - Loss: {avg_val_loss:.4f} - {val_loss_str}")
        
        # 早停检测
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, "best_aligner_model_checkpoint.pt")
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # 加载最佳模型
    checkpoint = torch.load("best_aligner_model_checkpoint.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    return model


def align_embeddings_enhanced(graph_embeds, text_embeds, labels, latent_dim=256, num_epochs=50, 
                              batch_size=32, eval_interval=5):
    """增强型嵌入对齐函数，包括数据增强和更多评估"""
    print("开始图-文本嵌入对齐增强流程...")
    
    # 创建数据集
    dataset = torch.utils.data.TensorDataset(
        graph_embeds,
        text_embeds,
        labels
    )
    
    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建加权采样器处理类别不平衡
    train_labels = torch.tensor([dataset[i][2] for i in train_dataset.indices])
    sampler = WeightedRandomSampler(train_labels)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler(len(train_dataset)*2)
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    # 初始化增强型模型
    num_classes = len(torch.unique(labels))
    model = EnhancedMultiModalAligner(
        graph_dim=graph_embeds.shape[1],
        text_dim=text_embeds.shape[1],
        latent_dim=latent_dim,
        num_classes=num_classes
    )
    
    # 打印模型结构摘要
    print(f"模型结构: 图嵌入维度 {graph_embeds.shape[1]}, 文本嵌入维度 {text_embeds.shape[1]}")
    print(f"潜在空间维度: {latent_dim}, 类别数: {num_classes}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}, 可训练参数: {trainable_params:,}")
    
    # 训练模型
    print("开始训练增强型多模态对齐模型...")
    model = train_aligner_with_scheduler(
        model, train_loader, val_loader, 
        num_epochs=num_epochs,
        lr=0.001,
        weight_decay=1e-5
    )
    
    # 评估对齐效果
    print("\n评估对齐效果...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # 收集测试集预测
    test_preds = {"graph": [], "text": [], "fusion": []}
    test_labels = []
    test_graph_embeds = []
    test_text_embeds = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            graph_embed, text_embed, labels = [b.to(device) for b in batch]
            outputs = model(graph_embed, text_embed)
            
            # 收集预测结果
            if model.num_classes:
                test_preds["graph"].extend(torch.argmax(outputs["graph_preds"], dim=1).cpu().numpy())
                test_preds["text"].extend(torch.argmax(outputs["text_preds"], dim=1).cpu().numpy())
                test_preds["fusion"].extend(torch.argmax(outputs["fusion_preds"], dim=1).cpu().numpy())
            
            # 收集嵌入和标签
            test_graph_embeds.append(outputs["graph_latent"].cpu())
            test_text_embeds.append(outputs["text_latent"].cpu())
            test_labels.extend(labels.cpu().numpy())
    
    # 合并收集的嵌入
    test_graph_embeds = torch.cat(test_graph_embeds)
    test_text_embeds = torch.cat(test_text_embeds)
    test_labels = np.array(test_labels)
    
    # 评估分类性能
    if model.num_classes:
        print("\n分类性能:")
        for modal, preds in test_preds.items():
            accuracy = accuracy_score(test_labels, preds)
            print(f"{modal}模态准确率: {accuracy:.4f}")
    
    # 计算跨模态检索性能
    print("\n跨模态检索性能:")
    calc_retrieval_metrics(test_graph_embeds, test_text_embeds, np.array(test_labels))
    
    # 保存对齐后的嵌入
    print("\n生成并保存所有节点的对齐嵌入...")
    aligned_graph_embeds, aligned_text_embeds = get_aligned_embeddings(model, graph_embeds, text_embeds)
    
    # 保存模型和嵌入
    torch.save(model.state_dict(), "enhanced_aligner_model.pt")
    torch.save(aligned_graph_embeds, "enhanced_aligned_graph_embeddings.pt")
    torch.save(aligned_text_embeds, "enhanced_aligned_text_embeddings.pt")
    
    # 保存融合嵌入（图+文本结合）
    with torch.no_grad():
        device = next(model.parameters()).device
        batch_size = 128
        all_fused = []
        
        for i in range(0, len(graph_embeds), batch_size):
            g_batch = graph_embeds[i:i+batch_size].to(device)
            t_batch = text_embeds[i:i+batch_size].to(device)
            outputs = model(g_batch, t_batch)
            # 使用联合表示作为融合嵌入
            all_fused.append(outputs["joint_latent"].cpu())
        
        fused_embeddings = torch.cat(all_fused)
        torch.save(fused_embeddings, "enhanced_fused_embeddings.pt")
    
    print(f"\n增强型对齐完成！模型和对齐后的嵌入已保存。")
    print(f"测试集大小: {len(test_labels)}")
    print(f"全部对齐嵌入形状: 图 {aligned_graph_embeds.shape}, 文本 {aligned_text_embeds.shape}")
    print(f"融合嵌入形状: {fused_embeddings.shape}")
    
    return model, {
        "aligned_graph_embeds": aligned_graph_embeds,
        "aligned_text_embeds": aligned_text_embeds,
        "fused_embeddings": fused_embeddings,
        "test_performance": {
            "graph_accuracy": accuracy_score(test_labels, test_preds["graph"]) if model.num_classes else None,
            "text_accuracy": accuracy_score(test_labels, test_preds["text"]) if model.num_classes else None,
            "fusion_accuracy": accuracy_score(test_labels, test_preds["fusion"]) if model.num_classes else None,
        }
    }


def get_aligned_embeddings(model, graph_embeds, text_embeds, batch_size=128):
    """获取所有节点的对齐嵌入"""
    device = next(model.parameters()).device
    model.eval()
    
    all_graph_latent = []
    all_text_latent = []
    
    with torch.no_grad():
        for i in range(0, len(graph_embeds), batch_size):
            g_batch = graph_embeds[i:i+batch_size].to(device)
            t_batch = text_embeds[i:i+batch_size].to(device)
            
            outputs = model(g_batch, t_batch)
            
            all_graph_latent.append(outputs["graph_latent"].cpu())
            all_text_latent.append(outputs["text_latent"].cpu())
    
    all_graph_latent = torch.cat(all_graph_latent)
    all_text_latent = torch.cat(all_text_latent)
    
    return all_graph_latent, all_text_latent


def calc_retrieval_metrics(graph_embeds, text_embeds, labels=None, Ks=[1, 5, 10]):
    """计算更详细的跨模态检索指标"""
    # 计算相似度矩阵
    sim_matrix = torch.matmul(graph_embeds, text_embeds.transpose(0, 1))
    
    # 计算Recall@K
    recall = {}
    for k in Ks:
        # 图->文本检索
        topk_indices = torch.topk(sim_matrix, k, dim=1)[1]
        if labels is None:
            # 使用索引匹配
            correct = torch.sum(topk_indices == torch.arange(len(sim_matrix)).unsqueeze(1)).item()
            recall[f'G2T_R@{k}'] = correct / len(sim_matrix)
        else:
            # 使用标签匹配
            correct = 0
            for i, topk_idx in enumerate(topk_indices):
                target_label = labels[i]
                retrieved_labels = labels[topk_idx.cpu().numpy()]
                if target_label in retrieved_labels:
                    correct += 1
            recall[f'G2T_R@{k}'] = correct / len(sim_matrix)
        
        # 文本->图检索
        topk_indices = torch.topk(sim_matrix.t(), k, dim=1)[1]
        if labels is None:
            correct = torch.sum(topk_indices == torch.arange(len(sim_matrix)).unsqueeze(1)).item()
            recall[f'T2G_R@{k}'] = correct / len(sim_matrix)
        else:
            correct = 0
            for i, topk_idx in enumerate(topk_indices):
                target_label = labels[i]
                retrieved_labels = labels[topk_idx.cpu().numpy()]
                if target_label in retrieved_labels:
                    correct += 1
            recall[f'T2G_R@{k}'] = correct / len(sim_matrix)
    
    # 计算平均排名
    if labels is None:
        # 计算每个样本的真正匹配在排名中的位置
        ranks_g2t = []
        ranks_t2g = []
        
        for i in range(len(sim_matrix)):
            # 找出第i个图嵌入与文本嵌入i的匹配排名
            sorted_indices = torch.argsort(sim_matrix[i], descending=True)
            rank = int((sorted_indices == i).nonzero(as_tuple=True)[0].item()) + 1
            ranks_g2t.append(rank)
            
            # 找出第i个文本嵌入与图嵌入i的匹配排名
            sorted_indices = torch.argsort(sim_matrix[:, i], descending=True)
            rank = int((sorted_indices == i).nonzero(as_tuple=True)[0].item()) + 1
            ranks_t2g.append(rank)
        
        # 计算平均排名和中位数排名
        recall['G2T_MeanRank'] = np.mean(ranks_g2t)
        recall['G2T_MedianRank'] = np.median(ranks_g2t)
        recall['T2G_MeanRank'] = np.mean(ranks_t2g)
        recall['T2G_MedianRank'] = np.median(ranks_t2g)
    
    # 打印指标
    print("\n跨模态检索性能:")
    for k, v in recall.items():
        print(f"{k}: {v:.4f}")
    
    return recall


if __name__ == "__main__":
    print("加载图嵌入和文本嵌入...")
    
    try:
        # 从PT文件加载数据
        graph_embeds = torch.load('saved_models/cora_node/node_embeddings_GAT.pt')
        text_embeds = torch.load('data_preprocess/Dataset/cora/emb/cora_fused_fused_node.pt')
        labels = torch.load('data_preprocess/Dataset/cora/emb/numeric_labels.pt')
        print(f"成功加载数据：图嵌入 {graph_embeds.shape}，文本嵌入 {text_embeds.shape}，标签 {labels.shape}")
    except:
        print("未找到嵌入数据文件，生成示例数据进行演示...")
        num_samples = 500
        num_classes = 5
        graph_dim = 64
        text_dim = 128
        
        # 生成示例嵌入
        graph_embeds = torch.randn(num_samples, graph_dim)
        text_embeds = torch.randn(num_samples, text_dim)
        labels = torch.randint(0, num_classes, (num_samples,))
    
    # 运行增强型对齐
    model, results = align_embeddings_enhanced(
        graph_embeds, 
        text_embeds, 
        labels, 
        latent_dim=256, 
        num_epochs=50,
        batch_size=32
    )