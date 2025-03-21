import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import argparse
import os
from aligner import EnhancedMultiModalAligner
from dataset import create_data_loaders, WeightedRandomSampler
from evaluation import calc_retrieval_metrics, evaluate_model, get_aligned_embeddings
from training import train_aligner_with_scheduler

def align_embeddings_enhanced(graph_embeds, text_embeds, labels, output_dir='output', 
                             latent_dim=256, num_epochs=50, batch_size=32, eval_interval=5):
    """增强型嵌入对齐函数，包括数据增强和更多评估"""
    print("开始图-文本嵌入对齐增强流程...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保所有输入的数据类型一致
    graph_embeds = graph_embeds.to(torch.float32)
    text_embeds = text_embeds.to(torch.float32)

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        graph_embeds, text_embeds, labels, batch_size=batch_size
    )
    
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
    
    # 评估模型
    print("\n评估对齐效果...")
    eval_results = evaluate_model(model, test_loader)
    
    # 保存对齐后的嵌入
    print("\n生成并保存所有节点的对齐嵌入...")
    aligned_graph_embeds, aligned_text_embeds = get_aligned_embeddings(model, graph_embeds, text_embeds)
    
    # 保存模型和嵌入（使用output_dir路径）
    torch.save(model.state_dict(), os.path.join(output_dir, "enhanced_aligner_model.pt"))
    torch.save(aligned_graph_embeds, os.path.join(output_dir, "enhanced_aligned_graph_embeddings.pt"))
    torch.save(aligned_text_embeds, os.path.join(output_dir, "enhanced_aligned_text_embeddings.pt"))
    
    
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
        torch.save(fused_embeddings, os.path.join(output_dir, "enhanced_fused_embeddings.pt"))
    
    print(f"\n增强型对齐完成！模型和对齐后的嵌入已保存。")
    print(f"测试集大小: {len(eval_results['test_labels'])}")
    print(f"全部对齐嵌入形状: 图 {aligned_graph_embeds.shape}, 文本 {aligned_text_embeds.shape}")
    print(f"融合嵌入形状: {fused_embeddings.shape}")
    
    return model, {
        "aligned_graph_embeds": aligned_graph_embeds,
        "aligned_text_embeds": aligned_text_embeds,
        "fused_embeddings": fused_embeddings,
        "test_performance": eval_results
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced Multi-Modal Aligner')
    parser.add_argument('--graph_embeds_path', type=str, required=True, help='Path to graph embeddings')
    parser.add_argument('--text_embeds_path', type=str, required=True, help='Path to text embeddings')
    parser.add_argument('--labels_path', type=str, required=True, help='Path to labels')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of latent space')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Loading graph and text embeddings from {args.graph_embeds_path} and {args.text_embeds_path}...")
    
    # Load data from files
    graph_embeds = torch.load(args.graph_embeds_path)
    text_embeds = torch.load(args.text_embeds_path)
    labels = torch.load(args.labels_path)
    
    print(f"Successfully loaded data: Graph embeddings {graph_embeds.shape}, Text embeddings {text_embeds.shape}, Labels {labels.shape}")
    
    # Run enhanced alignment
    model, results = align_embeddings_enhanced(
        graph_embeds, 
        text_embeds, 
        labels,
        args.output_dir,
        latent_dim=args.latent_dim,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )
    
    print("Alignment process completed successfully!")

# python Alignment/alignment/main.py --graph_embeds_path saved_models/cora_node/node_embeddings_GAT.pt --text_embeds_path data_preprocess/Dataset/cora/emb/cora_fused_fused_node.pt --labels_path data_preprocess/Dataset/cora/emb/numeric_labels.pt --output_dir data_preprocess/Dataset/cora/alignment --latent_dim 256 --num_epochs 50 --batch_size 32 --learning_rate 0.001 --weight_decay 1e-5 --seed 42