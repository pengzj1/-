import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def get_aligned_embeddings(model, graph_embeds, text_embeds, batch_size=128):
    """获取所有节点的对齐嵌入"""
    device = next(model.parameters()).device
    model.eval()

    # 确保与模型一致的数据类型
    model_dtype = next(model.parameters()).dtype
    graph_embeds = graph_embeds.to(device=device, dtype=model_dtype)
    text_embeds = text_embeds.to(device=device, dtype=model_dtype)
    
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

def calc_retrieval_metrics(graph_embeds, text_embeds, labels=None, Ks=None):
    """计算更详细的跨模态检索指标"""
    # 设置默认K值
    if Ks is None:
        Ks = [1, 5, 10]
    
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

def evaluate_model(model, test_loader):
    """评估模型性能"""
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
    
    results = {
        "test_graph_embeds": test_graph_embeds,
        "test_text_embeds": test_text_embeds,
        "test_labels": test_labels
    }
    
    # 评估分类性能
    if model.num_classes:
        print("\n分类性能:")
        for modal, preds in test_preds.items():
            accuracy = accuracy_score(test_labels, preds)
            print(f"{modal}模态准确率: {accuracy:.4f}")
            results[f"{modal}_accuracy"] = accuracy
    
    # 计算跨模态检索性能
    retrieval_metrics = calc_retrieval_metrics(test_graph_embeds, test_text_embeds, test_labels)
    results["retrieval_metrics"] = retrieval_metrics
    
    return results