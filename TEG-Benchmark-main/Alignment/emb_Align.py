import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class MultiModalAligner(nn.Module):
    """多模态对齐模型：整合预测对齐与潜在空间对齐"""
    
    def __init__(self, graph_dim, text_dim, latent_dim=128, num_classes=None):
        """
        参数:
            graph_dim: 图嵌入维度
            text_dim: 文本嵌入维度
            latent_dim: 潜在空间维度
            num_classes: 分类任务的类别数量
        """
        super(MultiModalAligner, self).__init__()
        
        # 投影层：将不同模态映射到同一潜在空间
        self.graph_projector = nn.Sequential(
            nn.Linear(graph_dim, latent_dim*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim*2, latent_dim)
        )
        
        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, latent_dim*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim*2, latent_dim)
        )
        
        # 预测头：用于预测任务（如分类）
        if num_classes:
            self.graph_predictor = nn.Linear(latent_dim, num_classes)
            self.text_predictor = nn.Linear(latent_dim, num_classes)
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.temperature = 0.07  # 对比学习温度参数
    
    def forward(self, graph_embeds, text_embeds):
        """前向传播，返回潜在空间嵌入和预测结果"""
        # 投影到潜在空间
        graph_latent = self.graph_projector(graph_embeds)
        text_latent = self.text_projector(text_embeds)
        
        # L2归一化嵌入（对对比学习很重要）
        graph_latent_norm = nn.functional.normalize(graph_latent, p=2, dim=1)
        text_latent_norm = nn.functional.normalize(text_latent, p=2, dim=1)
        
        results = {
            "graph_latent": graph_latent,
            "text_latent": text_latent,
            "graph_latent_norm": graph_latent_norm,
            "text_latent_norm": text_latent_norm
        }
        
        # 如果有分类任务，计算预测结果
        if self.num_classes:
            results["graph_preds"] = self.graph_predictor(graph_latent)
            results["text_preds"] = self.text_predictor(text_latent)
        
        return results
    
    def compute_contrastive_loss(self, graph_latent, text_latent, labels=None):
        """计算对比学习损失（潜在空间对齐）"""
        batch_size = graph_latent.shape[0]
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(graph_latent, text_latent.transpose(0, 1)) / self.temperature
        
        # 如果有标签信息，可以用于构造更精确的正负样本对
        if labels is not None:
            pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            # 正样本对：相同标签的样本对
            # 负样本对：不同标签的样本对
            loss = 0
            for i in range(batch_size):
                pos_indices = torch.where(pos_mask[i] == 1)[0]
                if len(pos_indices) > 1:  # 至少有一个正样本（除自身外）
                    pos_logits = sim_matrix[i, pos_indices]
                    neg_logits = sim_matrix[i, pos_mask[i] == 0]
                    logits = torch.cat([pos_logits, neg_logits])
                    labels = torch.zeros(len(logits), dtype=torch.long, device=sim_matrix.device)
                    labels[:len(pos_logits)] = 1
                    loss += nn.CrossEntropyLoss()(logits.unsqueeze(0), labels.unsqueeze(0))
            return loss / batch_size
        else:
            # 无标签情况：假设每个嵌入与其对应位置的另一模态嵌入匹配
            labels = torch.arange(batch_size, device=sim_matrix.device)
            loss = nn.CrossEntropyLoss()(sim_matrix, labels) + nn.CrossEntropyLoss()(sim_matrix.t(), labels)
            return loss / 2
    
    def compute_prediction_loss(self, graph_preds, text_preds, labels, consistency_weight=0.5):
        """计算预测对齐损失"""
        # 任务损失（如分类交叉熵）
        graph_task_loss = nn.CrossEntropyLoss()(graph_preds, labels)
        text_task_loss = nn.CrossEntropyLoss()(text_preds, labels)
        
        # 预测一致性损失（MSE）
        consistency_loss = nn.MSELoss()(
            nn.functional.softmax(graph_preds, dim=1),
            nn.functional.softmax(text_preds, dim=1)
        )
        
        # 总预测损失
        pred_loss = graph_task_loss + text_task_loss + consistency_weight * consistency_loss
        return pred_loss, {
            "graph_task_loss": graph_task_loss.item(),
            "text_task_loss": text_task_loss.item(),
            "consistency_loss": consistency_loss.item()
        }
    
    def compute_total_loss(self, batch, pred_weight=1.0, contrastive_weight=1.0):
        """计算总损失：预测对齐 + 潜在空间对齐"""
        graph_embeds, text_embeds, labels = batch
        
        # 前向传播
        outputs = self.forward(graph_embeds, text_embeds)
        
        losses = {}
        total_loss = 0
        
        # 潜在空间对齐损失（对比学习）
        contrastive_loss = self.compute_contrastive_loss(
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
                labels
            )
            losses.update(pred_losses)
            total_loss += pred_weight * pred_loss
        
        return total_loss, losses, outputs


def train_aligner(model, train_loader, val_loader, num_epochs=20, lr=0.001):
    """训练多模态对齐模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            # 将数据移到设备上
            batch = [item.to(device) for item in batch]
            
            optimizer.zero_grad()
            loss, losses, _ = model.compute_total_loss(batch)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = [item.to(device) for item in batch]
                loss, _, _ = model.compute_total_loss(batch)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_aligner_model.pt")
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_curve.png')
    plt.close()
    
    return model


def evaluate_alignment(model, test_loader, device=None):
    """评估对齐效果"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    all_graph_embeds = []
    all_text_embeds = []
    all_labels = []
    graph_preds = []
    text_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = [item.to(device) for item in batch]
            graph_embed, text_embed, labels = batch
            
            outputs = model(graph_embed, text_embed)
            
            all_graph_embeds.append(outputs["graph_latent"].cpu().numpy())
            all_text_embeds.append(outputs["text_latent"].cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            if model.num_classes:
                graph_preds.append(torch.argmax(outputs["graph_preds"], dim=1).cpu().numpy())
                text_preds.append(torch.argmax(outputs["text_preds"], dim=1).cpu().numpy())
    
    all_graph_embeds = np.vstack(all_graph_embeds)
    all_text_embeds = np.vstack(all_text_embeds)
    all_labels = np.concatenate(all_labels)
    
    # 可视化对齐效果
    visualize_alignments(all_graph_embeds, all_text_embeds, all_labels)
    
    # 评估分类准确率
    if model.num_classes:
        graph_preds = np.concatenate(graph_preds)
        text_preds = np.concatenate(text_preds)
        
        graph_acc = accuracy_score(all_labels, graph_preds)
        text_acc = accuracy_score(all_labels, text_preds)
        consistency = np.mean(graph_preds == text_preds)
        
        print(f"图模态分类准确率: {graph_acc:.4f}")
        print(f"文本模态分类准确率: {text_acc:.4f}")
        print(f"预测一致性: {consistency:.4f}")
    
    # 计算跨模态检索精度
    sim_matrix = cosine_similarity(all_graph_embeds, all_text_embeds)
    
    # 计算Recall@K
    Ks = [1, 5, 10]
    recall_at_k = {}
    
    for k in Ks:
        # 图->文本检索
        correct = 0
        for i in range(len(sim_matrix)):
            top_k = np.argsort(sim_matrix[i])[-k:]
            if i in top_k:
                correct += 1
        recall_at_k[f'G2T_R@{k}'] = correct / len(sim_matrix)
        
        # 文本->图检索
        correct = 0
        for i in range(len(sim_matrix[0])):
            top_k = np.argsort(sim_matrix[:, i])[-k:]
            if i in top_k:
                correct += 1
        recall_at_k[f'T2G_R@{k}'] = correct / len(sim_matrix[0])
    
    print("\n跨模态检索性能:")
    for k, v in recall_at_k.items():
        print(f"{k}: {v:.4f}")
    
    return {
        "graph_embeds": all_graph_embeds,
        "text_embeds": all_text_embeds,
        "labels": all_labels,
        "retrieval_metrics": recall_at_k
    }


def visualize_alignments(graph_embeds, text_embeds, labels):
    """可视化潜在空间中的对齐效果"""
    # 使用t-SNE降维进行可视化
    combined = np.vstack([graph_embeds, text_embeds])
    tsne = TSNE(n_components=2, random_state=42)
    combined_2d = tsne.fit_transform(combined)
    
    graph_2d = combined_2d[:len(graph_embeds)]
    text_2d = combined_2d[len(graph_embeds):]
    
    plt.figure(figsize=(12, 10))
    
    # 绘制图嵌入
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(
            graph_2d[indices, 0], 
            graph_2d[indices, 1], 
            marker='o', 
            label=f'Graph Class {label}'
        )
    
    # 绘制文本嵌入
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(
            text_2d[indices, 0], 
            text_2d[indices, 1], 
            marker='x', 
            label=f'Text Class {label}'
        )
    
    plt.title('t-SNE Visualization of Aligned Graph and Text Embeddings')
    plt.legend()
    plt.savefig('alignment_visualization.png')
    plt.close()


def prepare_for_llm(graph_embeds, text_embeds, labels, data_descriptions):
    """准备数据给大语言模型，展示对齐效果"""
    # 计算同类别图-文嵌入相似度
    avg_intra_sim = 0
    for label in np.unique(labels):
        indices = labels == label
        graph_cls = graph_embeds[indices]
        text_cls = text_embeds[indices]
        
        if len(graph_cls) > 0 and len(text_cls) > 0:
            sim_matrix = cosine_similarity(graph_cls, text_cls)
            avg_intra_sim += np.mean(sim_matrix)
    
    avg_intra_sim /= len(np.unique(labels))
    
    # 随机采样一些样本进行展示
    np.random.seed(42)
    sample_indices = np.random.choice(len(labels), min(5, len(labels)), replace=False)
    
    samples = []
    for i in sample_indices:
        graph_sim = cosine_similarity([graph_embeds[i]], text_embeds)[0]
        text_sim = cosine_similarity([text_embeds[i]], graph_embeds)[0]
        
        # 找到最相似的前3个
        graph_top3 = np.argsort(graph_sim)[-3:][::-1]
        text_top3 = np.argsort(text_sim)[-3:][::-1]
        
        samples.append({
            "id": i,
            "label": int(labels[i]),
            "description": data_descriptions[i] if data_descriptions else f"Sample {i}",
            "graph_top3_matches": [int(idx) for idx in graph_top3],
            "text_top3_matches": [int(idx) for idx in text_top3],
            "correct_graph_match": i in graph_top3,
            "correct_text_match": i in text_top3,
        })
    
    # 构建LLM查询结果
    llm_query = {
        "alignment_quality": {
            "avg_intra_class_similarity": float(avg_intra_sim),
        },
        "sample_matches": samples
    }
    
    return llm_query


# 主函数：演示如何使用对齐框架
def demo_alignment(graph_embeds, text_embeds, labels, data_descriptions=None, latent_dim=128, num_epochs=20):
    """演示如何使用多模态对齐框架"""
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(graph_embeds),
        torch.FloatTensor(text_embeds),
        torch.LongTensor(labels)
    )
    
    # 划分训练集、验证集、测试集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    # 初始化模型
    num_classes = len(np.unique(labels))
    model = MultiModalAligner(
        graph_dim=graph_embeds.shape[1],
        text_dim=text_embeds.shape[1],
        latent_dim=latent_dim,
        num_classes=num_classes
    )
    
    # 训练模型
    print("开始训练多模态对齐模型...")
    model = train_aligner(model, train_loader, val_loader, num_epochs=num_epochs)
    
    # 评估对齐效果
    print("\n评估对齐效果...")
    eval_results = evaluate_alignment(model, test_loader)
    
    # 准备数据给LLM
    if data_descriptions is None:
        data_descriptions = [f"样本{i}" for i in range(len(labels))]
    
    llm_input = prepare_for_llm(
        eval_results["graph_embeds"], 
        eval_results["text_embeds"], 
        eval_results["labels"],
        data_descriptions
    )
    
    print("\n已准备好对齐结果数据，可传递给大语言模型进行评估。")
    return model, eval_results, llm_input


if __name__ == "__main__":
    print("加载已有的图嵌入和文本嵌入...")
    # 这里应该替换为您已有的嵌入数据
    # 以下仅为示例，请替换为实际数据
    try:
        # 尝试从文件加载数据
        graph_embeds = np.load('graph_embeddings.npy')
        text_embeds = np.load('text_embeddings.npy')
        labels = np.load('labels.npy')
        print(f"成功加载数据：图嵌入 {graph_embeds.shape}，文本嵌入 {text_embeds.shape}，标签 {labels.shape}")
    except:
        # 如果没有现成数据，生成一些示例数据进行演示
        print("未找到嵌入数据文件，生成示例数据进行演示...")
        num_samples = 500
        num_classes = 5
        graph_dim = 64
        text_dim = 128
        
        # 生成示例嵌入
        graph_embeds = np.random.randn(num_samples, graph_dim)
        text_embeds = np.random.randn(num_samples, text_dim)
        labels = np.random.randint(0, num_classes, size=num_samples)
        
        # 为了模拟真实情况，让相同类的嵌入更相似
        for i in range(num_samples):
            class_idx = labels[i]
            # 添加类特定的信号
            class_signal_graph = np.random.randn(graph_dim) * 2
            class_signal_text = np.random.randn(text_dim) * 2
            
            graph_embeds[i] += class_signal_graph
            text_embeds[i] += class_signal_text
    
    # 运行对齐演示
    model, eval_results, llm_input = demo_alignment(graph_embeds, text_embeds, labels)
    
    # 保存对齐后的嵌入
    np.save('aligned_graph_embeddings.npy', eval_results["graph_embeds"])
    np.save('aligned_text_embeddings.npy', eval_results["text_embeds"])
    
    print("\n对齐完成！可以将对齐后的嵌入提供给大语言模型进行进一步分析。")

# 代码详细解释
# 1. 多模态对齐模型 (MultiModalAligner)
# 这个模型结合了预测对齐和潜在空间对齐两种策略：

# 潜在空间对齐:

# 使用投影层将图嵌入和文本嵌入映射到同一维度的潜在空间
# 通过对比学习损失函数强制相同类别的图-文嵌入在潜在空间中靠近
# 采用温度系数调整的余弦相似度来计算嵌入间的相似度
# 预测对齐:

# 为图嵌入和文本嵌入分别设计分类器
# 通过任务损失优化两种模态的分类性能
# 通过一致性损失强制两种模态的预测结果相似
# 2. 训练流程
# 训练过程包括：

# 使用Adam优化器和学习率调度器
# 同时优化潜在空间对齐损失和预测对齐损失
# 保存验证损失最低的模型
# 3. 评估方法
# 评估对齐效果的方式包括：

# 分类准确率：评估每种模态的分类性能
# 预测一致性：衡量两种模态预测结果的一致程度
# 跨模态检索性能：计算Recall@K指标
# 可视化：使用t-SNE降维可视化对齐效果
# 4. 与大语言模型集成
# 将对齐后的嵌入及其评估结果整理成结构化数据
# 提供样本匹配和相似度信息
# 便于大语言模型评估对齐效果并提供分析
# 实际应用价值
# 知识融合：通过将图结构知识与文本语义知识对齐，可以提高模型对复杂场景的理解能力。

# 跨模态检索：实现基于文本查询图数据或基于图结构查询相关文本的功能。

# 增强预测：利用多模态信息互补性，提高下游任务如分类、推荐等的准确率。

# 大语言模型增强：通过将对齐后的多模态知识注入到大语言模型中，扩展其在结构化数据上的推理能力。

# 该代码实现了一个完整的多模态对齐框架，可以根据实际数据需求进行适当调整。通过评估指标和可视化图表，可以直观地了解对齐效果，并可将结果提供给大语言模型进行进一步分析和应用。
