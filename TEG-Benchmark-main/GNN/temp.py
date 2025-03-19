# 添加自适应损失加权模块
class UncertaintyWeighting(torch.nn.Module):
    """
    基于任务不确定性的自适应损失加权
    论文: Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
    """
    def __init__(self, num_tasks=3):
        super(UncertaintyWeighting, self).__init__()
        # 初始化对数方差参数，使用log是为了保证权重为正数
        self.log_vars = torch.nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses):
        """
        计算加权损失
        Args:
            losses: 列表，包含每个任务的损失 [node_loss, link_loss, subgraph_loss]
        Returns:
            total_loss: 加权总损失
            weights: 每个任务的权重
        """
        weights = []
        weighted_losses = []
        
        for i, loss in enumerate(losses):
            # 计算权重：1/exp(log_var)
            precision = torch.exp(-self.log_vars[i])
            # 加权损失: loss * precision + log_var/2
            weighted_loss = precision * loss + 0.5 * self.log_vars[i]
            weights.append(precision.item())
            weighted_losses.append(weighted_loss)
            
        total_loss = sum(weighted_losses)
        return total_loss, weights


# 改进的训练函数，使用自适应损失加权
def train_multitask_adaptive(model, node_classifier, link_predictor, subgraph_classifier, 
                             node_loader, link_loader, subgraph_loader, 
                             optimizer, node_criterion, subgraph_criterion, 
                             uncertainty_weighting, device, args):
    model.train()
    node_classifier.train()
    link_predictor.train()
    subgraph_classifier.train()

    total_loss = 0
    node_loss_sum = 0
    link_loss_sum = 0
    subgraph_loss_sum = 0
    
    # 批次计数器
    batch_count = 0
    weights_sum = [0, 0, 0]
    
    # 节点分类任务处理
    print("Training node classification task...")
    for batch in tqdm.tqdm(node_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        
        # 使用记忆高效的方式计算节点嵌入
        h = model(batch.x, batch.adj_t)
        pred = node_classifier(h[:batch.batch_size])
        node_loss = node_criterion(pred, batch.y[:batch.batch_size])
        
        # 存储中间结果，不立即反向传播
        node_loss_batch = node_loss.item()
        node_loss_sum += node_loss_batch
        
        # 处理子图分类任务（共享同一批次的节点嵌入）
        if hasattr(args, 'joint_batching') and args.joint_batching:
            # 假设我们有子图分类的批次
            if batch_count < len(subgraph_loader):
                subgraph_batch = list(subgraph_loader)[batch_count]
                batch_subgraphs, batch_labels = subgraph_batch
                
                # 计算子图嵌入
                subgraph_embeddings = []
                for subgraph in batch_subgraphs:
                    subgraph = subgraph.to(device)
                    # 我们需要共享计算的节点嵌入
                    # 这里需要找到子图中的节点在当前批次中的索引
                    # 简化版本，实际实现可能需要更复杂的处理
                    subgraph_embedding = torch.mean(h[subgraph], dim=0)
                    subgraph_embeddings.append(subgraph_embedding)
                
                if subgraph_embeddings:
                    subgraph_embeddings = torch.stack(subgraph_embeddings)
                    subgraph_preds = subgraph_classifier(subgraph_embeddings)
                    subgraph_loss = subgraph_criterion(subgraph_preds, batch_labels.to(device))
                    subgraph_loss_batch = subgraph_loss.item()
                else:
                    subgraph_loss = torch.tensor(0.0, device=device)
                    subgraph_loss_batch = 0
            else:
                subgraph_loss = torch.tensor(0.0, device=device)
                subgraph_loss_batch = 0
        else:
            subgraph_loss = torch.tensor(0.0, device=device)
            subgraph_loss_batch = 0
        
        subgraph_loss_sum += subgraph_loss_batch
        
        # 链接预测任务处理
        link_loss = torch.tensor(0.0, device=device)
        link_loss_batch = 0
        
        # 计算自适应加权的损失
        losses = [node_loss, link_loss, subgraph_loss]
        loss, weights = uncertainty_weighting(losses)
        
        # 累加权重用于记录
        for i, w in enumerate(weights):
            weights_sum[i] += w
        
        # 反向传播和优化
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_val)
        
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
    
    # 独立处理链接预测任务
    print("Training link prediction task...")
    for batch in tqdm.tqdm(link_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        
        # 使用分块处理大型图以节省内存
        if batch.x.size(0) > args.chunk_size and args.use_chunking:
            h = process_in_chunks(model, batch.x, batch.adj_t, chunk_size=args.chunk_size)
        else:
            h = model(batch.x, batch.adj_t)
        
        src = batch.edge_label_index.t()[:, 0]
        dst = batch.edge_label_index.t()[:, 1]
        
        pos_out = link_predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        
        # 负采样
        dst_neg = torch.randint(0, h.size(0), src.size(), dtype=torch.long, device=h.device)
        neg_out = link_predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        
        link_loss = pos_loss + neg_loss
        link_loss_batch = link_loss.item()
        link_loss_sum += link_loss_batch
        
        # 使用零张量替代其他任务的损失
        node_loss = torch.tensor(0.0, device=device)
        subgraph_loss = torch.tensor(0.0, device=device)
        
        # 计算自适应加权的损失
        losses = [node_loss, link_loss, subgraph_loss]
        loss, weights = uncertainty_weighting(losses)
        
        # 累加权重用于记录
        for i, w in enumerate(weights):
            weights_sum[i] += w
        
        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_val)
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
    
    # 独立处理子图分类任务(如果没有在节点分类中联合处理)
    if not (hasattr(args, 'joint_batching') and args.joint_batching):
        print("Training subgraph classification task...")
        # 为子图分类预先计算节点嵌入（分批处理以提高内存效率）
        if args.use_chunking:
            h_all = compute_embeddings_in_chunks(model, node_loader.data, device, chunk_size=args.chunk_size)
        else:
            with torch.no_grad():
                data = Data(x=node_loader.data.x, adj_t=node_loader.data.adj_t)
                h_all = model(data.x.to(device), data.adj_t.to(device))
        
        for batch_subgraphs, batch_labels in tqdm.tqdm(subgraph_loader):
            optimizer.zero_grad()
            
            # 计算每个子图的嵌入
            subgraph_embeddings = []
            for subgraph in batch_subgraphs:
                subgraph = subgraph.to(device)
                # 获取子图节点的嵌入
                subgraph_node_embeddings = h_all[subgraph]
                # 计算子图嵌入（使用平均池化）
                subgraph_embedding = torch.mean(subgraph_node_embeddings, dim=0)
                subgraph_embeddings.append(subgraph_embedding)
            
            # 将子图嵌入堆叠成批次
            subgraph_embeddings = torch.stack(subgraph_embeddings)
            
            # 子图分类
            subgraph_preds = subgraph_classifier(subgraph_embeddings)
            subgraph_loss = subgraph_criterion(subgraph_preds, batch_labels.to(device))
            subgraph_loss_batch = subgraph_loss.item()
            subgraph_loss_sum += subgraph_loss_batch
            
            # 使用零张量替代其他任务的损失
            node_loss = torch.tensor(0.0, device=device)
            link_loss = torch.tensor(0.0, device=device)
            
            # 计算自适应加权的损失
            losses = [node_loss, link_loss, subgraph_loss]
            loss, weights = uncertainty_weighting(losses)
            
            # 累加权重用于记录
            for i, w in enumerate(weights):
                weights_sum[i] += w
            
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_val)
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
    
    # 计算平均损失
    avg_batch_count = max(batch_count, 1)  # 防止除零错误
    avg_node_loss = node_loss_sum / avg_batch_count
    avg_link_loss = link_loss_sum / avg_batch_count
    avg_subgraph_loss = subgraph_loss_sum / avg_batch_count
    
    # 计算平均权重
    avg_weights = [w / avg_batch_count for w in weights_sum]
    
    # 返回训练结果
    return {
        "total_loss": total_loss / avg_batch_count,
        "node_loss": avg_node_loss,
        "link_loss": avg_link_loss,
        "subgraph_loss": avg_subgraph_loss,
        "node_weight": avg_weights[0],
        "link_weight": avg_weights[1],
        "subgraph_weight": avg_weights[2]
    }


# 内存高效的块处理函数
def process_in_chunks(model, x, adj_t, chunk_size=10000):
    """
    分块处理大型图数据，减少内存消耗
    """
    model.eval()  # 设置为评估模式，避免批归一化等影响结果
    n = x.size(0)
    h = torch.zeros(n, model.hidden_channels, device=x.device)
    
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        # 获取第i到end个节点的邻接矩阵行
        chunk_adj = adj_t[i:end]
        # 计算这些节点的特征
        with torch.no_grad():  # 避免累积梯度
            chunk_out = model(x, chunk_adj)
        h[i:end] = chunk_out[i:end]
    
    model.train()  # 恢复训练模式
    return h


# 分块计算整图嵌入
def compute_embeddings_in_chunks(model, data, device, chunk_size=10000):
    """
    分批计算整个图的节点嵌入，以处理大型图
    """
    model.eval()
    n = data.x.size(0)
    h = torch.zeros(n, model.hidden_channels, device=device)
    
    # 将数据分成多个块
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        # 创建一个只包含当前块节点的子图
        # 注意：这里简化处理，实际上可能需要更复杂的子图抽取
        node_mask = torch.zeros(n, dtype=torch.bool)
        node_mask[i:end] = True
        
        # 获取子图的邻接矩阵
        row, col = data.adj_t.coo()[:2]
        mask = node_mask[row] & node_mask[col]
        sub_adj = SparseTensor(row=row[mask], col=col[mask], 
                              value=data.adj_t.storage.value()[mask],
                              sparse_sizes=(n, n))
        
        # 计算嵌入
        with torch.no_grad():
            chunk_h = model(data.x.to(device), sub_adj.to(device))
            h[i:end] = chunk_h[i:end]
    
    model.train()
    return h


# 图嵌入生成函数
def generate_embeddings(model, data, device, args, save_path=None):
    """
    生成并保存图嵌入
    Args:
        model: 训练好的GNN模型
        data: 图数据
        device: 计算设备
        args: 参数
        save_path: 保存路径
    Returns:
        node_embeddings: 节点嵌入
    """
    print("Generating node embeddings...")
    
    # 针对大型图的内存高效计算
    if args.use_chunking and data.x.size(0) > args.chunk_size:
        node_embeddings = compute_embeddings_in_chunks(
            model, data, device, chunk_size=args.chunk_size
        )
    else:
        # 对于较小的图，一次性计算全部嵌入
        model.eval()
        with torch.no_grad():
            data_on_device = Data(x=data.x.to(device), adj_t=data.adj_t.to(device))
            node_embeddings = model(data_on_device.x, data_on_device.adj_t).cpu()
    
    # 可选：保存嵌入
    if save_path:
        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(node_embeddings, save_path)
        print(f"Node embeddings saved to {save_path} with shape {node_embeddings.shape}")
    
    return node_embeddings


# 更新主函数中的相关部分
def main():
    parser = argparse.ArgumentParser(description="Multi-Task Learning for GNNs with Adaptive Weighting")
    
    # 现有参数
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gnn_model", type=str, default="GAT")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--neg_len", type=str, default="5000")
    
    # 添加新参数
    parser.add_argument("--use_adaptive_weighting", action="store_true", 
                        help="Whether to use adaptive uncertainty weighting for multi-task learning")
    parser.add_argument("--node_weight", type=float, default=1.0,
                        help="Initial weight for node classification loss (only used if adaptive weighting is off)")
    parser.add_argument("--link_weight", type=float, default=1.0,
                        help="Initial weight for link prediction loss (only used if adaptive weighting is off)")
    parser.add_argument("--subgraph_weight", type=float, default=1.0,
                        help="Initial weight for subgraph classification loss (only used if adaptive weighting is off)")
    parser.add_argument("--min_subgraph_size", type=int, default=5,
                        help="Minimum size of subgraphs to use")
    parser.add_argument("--use_chunking", action="store_true",
                        help="Whether to use memory-efficient chunking for large graphs")
    parser.add_argument("--chunk_size", type=int, default=10000,
                        help="Size of chunks when processing large graphs")
    parser.add_argument("--joint_batching", action="store_true",
                        help="Process multiple tasks in the same batch when possible")
    parser.add_argument("--clip_grad", action="store_true",
                        help="Whether to clip gradients during training")
    parser.add_argument("--grad_clip_val", type=float, default=1.0,
                        help="Maximum norm of gradients if gradient clipping is enabled")
    parser.add_argument("--save_embeddings", action="store_true",
                        help="Whether to save the learned node embeddings")
    parser.add_argument("--embeddings_path", type=str, default=None,
                        help="Path to save the learned embeddings")
    
    # 其他已有参数...
    parser.add_argument("--use_PLM_node", type=str,
                        default="data/CSTAG/Photo/Feature/children_gpt_node.pt")
    parser.add_argument("--use_PLM_edge", type=str,
                        default="data/CSTAG/Photo/Feature/children_gpt_edge.pt")
    parser.add_argument("--path", type=str,
                        default="data/CSTAG/Photo/LinkPrediction/")
    parser.add_argument("--graph_path", type=str,
                        default="data/CSTAG/Photo/children.pkl")
    
    args = parser.parse_args()
    print(args)
    
    # 设置wandb
    wandb.init(config=args, project="multitask-gnn-adaptive", reinit=True)
    
    # 确保目录存在
    if not os.path.exists(f"{args.path}{args.neg_len}/"):
        os.makedirs(f"{args.path}{args.neg_len}/")
    
    # 设置设备
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # 加载图数据
    with open(f"{args.graph_path}", "rb") as file:
        graph = pickle.load(file)
    
    # 设置节点数量
    graph.num_nodes = len(graph.text_nodes)
    
    # 处理节点分类数据
    node_split, y = process_node_data(args, device, graph)
    
    # 处理链接预测数据
    edge_split = process_link_data(args, graph)
    
    # 加载节点和边特征
    x = torch.load(args.use_PLM_node).squeeze().float()
    
    # 处理边特征和邻接矩阵
    edge_index = edge_split["train"]["edge"].t()
    edge_feature = torch.load(args.use_PLM_edge)[
        edge_split["train"]["train_edge_feature_index"]
    ].float()
    
    adj_t = SparseTensor.from_edge_index(
        edge_index, edge_feature, sparse_sizes=(graph.num_nodes, graph.num_nodes)
    ).t()
    adj_t = adj_t.to_symmetric()
    
    # 处理子图分类数据
    subgraph_split, num_subgraph_classes = process_subgraph_data(args, edge_index, graph.num_nodes, device)
    
    # 创建节点分类数据加载器
    data_node = Data(x=x, adj_t=adj_t, y=y)
    node_train_loader, node_val_loader, node_test_loader = gen_node_loaders(
        args, data_node, node_split
    )
    
    # 创建链接预测数据加载器
    link_train_loader, link_val_loader, link_val_negative_loader, link_test_loader, link_test_negative_loader = gen_link_loaders(
        args, edge_split, x, edge_index, adj_t
    )
    
    link_dataloaders = {
        "train": link_train_loader,
        "valid": link_val_loader,
        "valid_negative": link_val_negative_loader,
        "test": link_test_loader,
        "test_negative": link_test_negative_loader,
    }
    
    # 创建子图分类数据加载器
    subgraph_train_loader = SubgraphLoader(
        subgraph_split["train"]["subgraphs"],
        subgraph_split["train"]["labels"],
        args.batch_size,
        shuffle=True
    )
    
    subgraph_val_loader = SubgraphLoader(
        subgraph_split["val"]["subgraphs"],
        subgraph_split["val"]["labels"],
        args.batch_size,
        shuffle=False
    )
    
    subgraph_test_loader = SubgraphLoader(
        subgraph_split["test"]["subgraphs"],
        subgraph_split["test"]["labels"],
        args.batch_size,
        shuffle=False
    )
    
    # 设置随机种子
    seed_everything(42)
    
    # 创建评估器
    link_evaluator = LinkEvaluator()
    
    # 准备模型和优化器
    for run in range(args.runs):
        print(f"\nRun {run + 1}/{args.runs}")
        
        # 创建GNN模型
        model = gen_model(args, x, edge_feature)
        model = model.to(device)
        
        # 创建节点分类器
        node_classifier = Classifier(args.hidden_channels, y.size(1))
        node_classifier = node_classifier.to(device)
        
        # 创建链接预测器
        link_predictor = LinkPredictor(
            args.hidden_channels, args.hidden_channels, 1, args.num_layers, args.dropout
        )
        link_predictor = link_predictor.to(device)
        
        # 创建子图分类器
        subgraph_classifier = SubgraphClassifier(args.hidden_channels, num_subgraph_classes)
        subgraph_classifier = subgraph_classifier.to(device)
        
        # 创建不确定性加权模块（如果启用）
        if args.use_adaptive_weighting:
            uncertainty_weighting = UncertaintyWeighting(num_tasks=3)
            uncertainty_weighting = uncertainty_weighting.to(device)
            
            # 添加到优化器
            optimizer = torch.optim.Adam(
                list(model.parameters()) + 
                list(node_classifier.parameters()) + 
                list(link_predictor.parameters()) +
                list(subgraph_classifier.parameters()) +
                list(uncertainty_weighting.parameters()),
                lr=args.lr
            )
        else:
            # 使用固定权重
            uncertainty_weighting = None
            optimizer = torch.optim.Adam(
                list(model.parameters()) + 
                list(node_classifier.parameters()) + 
                list(link_predictor.parameters()) +
                list(subgraph_classifier.parameters()),
                lr=args.lr
            )
        
        # 设置损失函数
        node_criterion = torch.nn.BCEWithLogitsLoss()
        subgraph_criterion = torch.nn.CrossEntropyLoss()
        
        # 训练循环
        best_val_f1 = best_val_hits = best_val_subgraph_acc = 0
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            
            # 训练多任务模型
            if args.use_adaptive_weighting:
                train_loss = train_multitask_adaptive(
                    model, node_classifier, link_predictor, subgraph_classifier,
                    node_train_loader, link_train_loader, subgraph_train_loader,
                    optimizer, node_criterion, subgraph_criterion, 
                    uncertainty_weighting, device, args
                )
                
                # 记录训练损失和权重
                wandb.log({
                    "Epoch": epoch,
                    "Train/Total_Loss": train_loss["total_loss"],
                    "Train/Node_Loss": train_loss["node_loss"],
                    "Train/Link_Loss": train_loss["link_loss"],
                    "Train/Subgraph_Loss": train_loss["subgraph_loss"],
                    "Train/Node_Weight": train_loss["node_weight"],
                    "Train/Link_Weight": train_loss["link_weight"],
                    "Train/Subgraph_Weight": train_loss["subgraph_weight"]
                })
            else:
                # 使用原有的训练函数
                train_loss = train_multitask(
                    model, node_classifier, link_predictor, subgraph_classifier,
                    node_train_loader, link_train_loader, subgraph_train_loader,
                    optimizer, node_criterion, subgraph_criterion, device, args
                )
                
                # 记录训练损失
                wandb.log({
                    "Epoch": epoch,
                    "Train/Total_Loss": train_loss["total_loss"],
                    "Train/Node_Loss": train_loss["node_loss"],
                    "Train/Link_Loss": train_loss["link_loss"],
                    "Train/Subgraph_Loss": train_loss["subgraph_loss"]
                })
            
            # 定期评估
            if epoch % args.eval_steps == 0:
                # 节点分类评估
                val_node_results = eval_node_classification(
                    model, node_classifier, node_val_loader, args, device
                )
                
                # 链接预测评估
                val_link_results = eval_link_prediction(
                    model, link_predictor, link_dataloaders, link_evaluator, device
                )
                
                # 子图分类评估
                val_subgraph_results = eval_subgraph_classification(
                    model, subgraph_classifier, subgraph_val_loader, device
                )
                
                # 记录验证结果
                print(f"Validation Node F1: {val_node_results['f1']:.4f}")
                print(f"Validation Link Hits@50: {val_link_results['Hits@50'][1]:.4f}")
                print(f"Validation Subgraph Accuracy: {val_subgraph_results['accuracy']:.4f}")
                
                wandb.log({
                    "Epoch": epoch,
                    "Val/Node_F1": val_node_results['f1'],
                    "Val/Link_Hits@10": val_link_results['Hits@10'][1],
                    "Val/Link_Hits@50": val_link_results['Hits@50'][1],
                    "Val/Link_Hits@100": val_link_results['Hits@100'][1],
                    "Val/Subgraph_Accuracy": val_subgraph_results['accuracy']
                })
                
                # 保存最佳模型
                if val_node_results['f1'] > best_val_f1:
                    best_val_f1 = val_node_results['f1']
                    torch.save(node_classifier.state_dict(), f"{args.path}/best_node_classifier.pt")
                
                if val_link_results['Hits@50'][1] > best_val_hits:
                    best_val_hits = val_link_results['Hits@50'][1]
                    torch.save(link_predictor.state_dict(), f"{args.path}/best_link_predictor.pt")
                
                if val_subgraph_results['accuracy'] > best_val_subgraph_acc:
                    best_val_subgraph_acc = val_subgraph_results['accuracy']
                    torch.save(subgraph_classifier.state_dict(), f"{args.path}/best_subgraph_classifier.pt")
                
                # 保存GNN模型（所有任务共享）
                torch.save(model.state_dict(), f"{args.path}/best_model.pt")
        
        # 加载最佳模型进行测试
        model.load_state_dict(torch.load(f"{args.path}/best_model.pt"))
        node_classifier.
