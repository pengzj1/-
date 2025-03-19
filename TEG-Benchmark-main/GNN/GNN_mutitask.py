import torch
import pickle
import tqdm
import argparse
import os
import wandb
import numpy as np
from community import community_louvain

from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch_geometric.utils import to_networkx
from torch.nn import Linear


class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


# 新增：子图分类器
class SubgraphClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(SubgraphClassifier, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels // 2)
        self.lin3 = Linear(hidden_channels // 2, out_channels)
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        
    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        x = self.lin3(x)
        return x


def gen_model(args, x, edge_feature):
    if args.gnn_model == "GAT":
        model = GAT(
            x.size(1),
            edge_feature.size(1),
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.heads,
            args.dropout,
        )
    elif args.gnn_model == "GraphTransformer":
        model = GraphTransformer(
            x.size(1),
            edge_feature.size(1),
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.dropout,
        )
    elif args.gnn_model == "GINE":
        model = GINE(
            x.size(1),
            edge_feature.size(1),
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.dropout,
        )
    elif args.gnn_model == "GeneralGNN":
        model = GeneralGNN(
            x.size(1),
            edge_feature.size(1),
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.dropout,
        )
    else:
        raise ValueError("Not implemented")
    return model


def process_node_data(args, device, data):
    num_nodes = len(data.text_nodes)
    data.num_nodes = num_nodes

    product_indices = torch.tensor(
        [i for i, label in enumerate(data.text_node_labels) if label != -1]
    ).long()
    product_labels = [label for label in data.text_node_labels if label != -1]
    mlb = MultiLabelBinarizer()
    product_binary_labels = mlb.fit_transform(product_labels)
    y = torch.zeros(num_nodes, product_binary_labels.shape[1]).float()
    y[product_indices] = torch.tensor(product_binary_labels).float()
    y = y.to(device)

    train_ratio = 1 - args.test_ratio - args.val_ratio
    val_ratio = args.val_ratio

    num_products = product_indices.shape[0]
    train_idx = product_indices[: int(num_products * train_ratio)]
    val_idx = product_indices[
        int(num_products * train_ratio) : int(num_products * (train_ratio + val_ratio))
    ]
    test_idx = product_indices[
        int(product_indices.shape[0] * (train_ratio + val_ratio)) :
    ]

    node_split = {"train": train_idx, "val": val_idx, "test": test_idx}
    return node_split, y


def process_link_data(args, graph):
    edge_split = split_edge(
        graph,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        path=args.path,
        neg_len=args.neg_len,
    )
    
    return edge_split


# 新增：处理子图数据
def process_subgraph_data(args, edge_index, num_nodes, device):
    # 将边索引转换为NetworkX图用于社区检测
    edge_list = edge_index.t().cpu().numpy()
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_list)
    
    # 使用Louvain算法检测社区
    partition = community_louvain.best_partition(G)
    
    # 将节点映射到社区
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)
    
    # 过滤小社区
    min_community_size = args.min_subgraph_size
    filtered_communities = {k: v for k, v in communities.items() if len(v) >= min_community_size}
    
    # 重新分配社区ID
    community_mapping = {old_id: new_id for new_id, old_id in enumerate(filtered_communities.keys())}
    
    # 创建子图数据
    subgraphs = []
    subgraph_labels = []
    
    # 假设子图标签是基于社区ID的，或者您可以为社区分配自定义标签
    for old_id, nodes in filtered_communities.items():
        subgraphs.append(torch.tensor(nodes, dtype=torch.long))
        subgraph_labels.append(community_mapping[old_id])
    
    # 将标签转换为张量并进行独热编码
    num_classes = len(filtered_communities)
    subgraph_labels = torch.tensor(subgraph_labels, dtype=torch.long).to(device)
    
    # 分割训练、验证和测试集
    num_subgraphs = len(subgraphs)
    indices = torch.randperm(num_subgraphs)
    
    train_ratio = 1 - args.test_ratio - args.val_ratio
    val_ratio = args.val_ratio
    
    train_size = int(num_subgraphs * train_ratio)
    val_size = int(num_subgraphs * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_subgraphs = [subgraphs[i] for i in train_indices]
    val_subgraphs = [subgraphs[i] for i in val_indices]
    test_subgraphs = [subgraphs[i] for i in test_indices]
    
    train_labels = subgraph_labels[train_indices]
    val_labels = subgraph_labels[val_indices]
    test_labels = subgraph_labels[test_indices]
    
    subgraph_split = {
        "train": {"subgraphs": train_subgraphs, "labels": train_labels},
        "val": {"subgraphs": val_subgraphs, "labels": val_labels},
        "test": {"subgraphs": test_subgraphs, "labels": test_labels}
    }
    
    return subgraph_split, num_classes


def gen_node_loaders(args, data, node_split):
    train_loader = NeighborLoader(
        data,
        input_nodes=node_split["train"],
        num_neighbors=[10, 10],
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=node_split["val"],
        num_neighbors=[10, 10],
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = NeighborLoader(
        data,
        input_nodes=node_split["test"],
        num_neighbors=[10, 10],
        batch_size=args.batch_size,
        shuffle=False,
    )
    
    return train_loader, val_loader, test_loader


def gen_link_loaders(args, edge_split, x, edge_index, adj_t):
    data = Data(x=x, adj_t=adj_t)
    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[20, 10],
        edge_label_index=edge_index,
        edge_label=torch.ones(edge_index.shape[1]),
        batch_size=args.batch_size,
        neg_sampling_ratio=0.0,
        shuffle=True,
    )

    val_edge_label_index = edge_split["valid"]["edge"].t()
    val_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[20, 10],
        edge_label_index=val_edge_label_index,
        edge_label=torch.ones(val_edge_label_index.shape[1]),
        batch_size=args.batch_size,
        neg_sampling_ratio=0.0,
        shuffle=False,
    )

    val_negative_edge_label_index = edge_split["valid"]["edge_neg"].t()
    val_negative_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[20, 10],
        edge_label_index=val_negative_edge_label_index,
        edge_label=torch.zeros(val_negative_edge_label_index.shape[1]),
        batch_size=args.batch_size,
        neg_sampling_ratio=0.0,
        shuffle=False,
    )

    test_edge_label_index = edge_split["test"]["edge"].t()
    test_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[20, 10],
        edge_label_index=test_edge_label_index,
        edge_label=torch.ones(test_edge_label_index.shape[1]),
        batch_size=args.batch_size,
        neg_sampling_ratio=0.0,
        shuffle=False,
    )

    test_negative_edge_label_index = edge_split["test"]["edge_neg"].t()
    test_negative_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[20, 10],
        edge_label_index=test_negative_edge_label_index,
        edge_label=torch.zeros(test_negative_edge_label_index.shape[1]),
        batch_size=args.batch_size,
        neg_sampling_ratio=0.0,
        shuffle=False,
    )

    return train_loader, val_loader, val_negative_loader, test_loader, test_negative_loader


# 新增：生成子图数据加载器
class SubgraphLoader:
    def __init__(self, subgraphs, labels, batch_size, shuffle=True):
        self.subgraphs = subgraphs
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(subgraphs)
        
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.n)
            subgraphs = [self.subgraphs[i] for i in indices]
            labels = self.labels[indices]
        else:
            subgraphs = self.subgraphs
            labels = self.labels
            
        for i in range(0, self.n, self.batch_size):
            batch_subgraphs = subgraphs[i:i+self.batch_size]
            batch_labels = labels[i:i+self.batch_size]
            yield batch_subgraphs, batch_labels
            
    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


def train_multitask(model, node_classifier, link_predictor, subgraph_classifier, 
                    node_loader, link_loader, subgraph_loader, 
                    optimizer, node_criterion, subgraph_criterion, device, args):
    model.train()
    node_classifier.train()
    link_predictor.train()
    subgraph_classifier.train()

    # 定义损失权重
    node_weight = args.node_weight
    link_weight = args.link_weight
    subgraph_weight = args.subgraph_weight

    total_loss = 0
    node_loss_sum = 0
    link_loss_sum = 0
    subgraph_loss_sum = 0
    
    # 节点分类训练
    print("Training node classification task...")
    for batch in tqdm.tqdm(node_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        h = model(batch.x, batch.adj_t)
        pred = node_classifier(h[:batch.batch_size])
        node_loss = node_criterion(pred, batch.y[:batch.batch_size])
        
        # 加权损失
        weighted_loss = node_weight * node_loss
        weighted_loss.backward()
        optimizer.step()
        
        node_loss_sum += node_loss.item()
        total_loss += weighted_loss.item()
    
    # 链接预测训练
    print("Training link prediction task...")
    for batch in tqdm.tqdm(link_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        h = model(batch.x, batch.adj_t)
        
        src = batch.edge_label_index.t()[:, 0]
        dst = batch.edge_label_index.t()[:, 1]
        
        pos_out = link_predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        
        # 随机采样负样本
        dst_neg = torch.randint(0, h.size(0), src.size(), dtype=torch.long, device=h.device)
        neg_out = link_predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        
        link_loss = pos_loss + neg_loss
        
        # 加权损失
        weighted_loss = link_weight * link_loss
        weighted_loss.backward()
        optimizer.step()
        
        link_loss_sum += link_loss.item()
        total_loss += weighted_loss.item()
    
    # 子图分类训练
    print("Training subgraph classification task...")
    with torch.no_grad():
        # 预先计算所有节点的嵌入
        data = Data(x=node_loader.data.x, adj_t=node_loader.data.adj_t)
        h_all = model(data.x.to(device), data.adj_t.to(device))
    
    for batch_subgraphs, batch_labels in tqdm.tqdm(subgraph_loader):
        optimizer.zero_grad()
        
        # 计算每个子图的嵌入
        subgraph_embeddings = []
        for subgraph in batch_subgraphs:
            # 子图中的节点索引
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
        subgraph_loss = subgraph_criterion(subgraph_preds, batch_labels)
        
        # 加权损失
        weighted_loss = subgraph_weight * subgraph_loss
        weighted_loss.backward()
        optimizer.step()
        
        subgraph_loss_sum += subgraph_loss.item()
        total_loss += weighted_loss.item()
    
    avg_node_loss = node_loss_sum / len(node_loader)
    avg_link_loss = link_loss_sum / len(link_loader)
    avg_subgraph_loss = subgraph_loss_sum / len(subgraph_loader)
    
    return {
        "total_loss": total_loss / (len(node_loader) + len(link_loader) + len(subgraph_loader)),
        "node_loss": avg_node_loss,
        "link_loss": avg_link_loss,
        "subgraph_loss": avg_subgraph_loss
    }


@torch.no_grad()
def eval_node_classification(model, node_classifier, loader, args, device):
    model.eval()
    node_classifier.eval()
    
    preds = []
    ground_truths = []
    
    for batch in tqdm.tqdm(loader):
        batch = batch.to(device)
        h = model(batch.x, batch.adj_t)
        pred = node_classifier(h[:batch.batch_size])
        ground_truth = batch.y[:batch.batch_size]
        preds.append(pred)
        ground_truths.append(ground_truth)
    
    preds = torch.cat(preds, dim=0).cpu().numpy()
    ground_truths = torch.cat(ground_truths, dim=0).cpu().numpy()
    
    y_label = (preds > args.threshold).astype(int)
    f1 = f1_score(ground_truths, y_label, average="weighted")
    
    results = {"f1": f1}
    
    data_type = args.graph_path.split("/")[-1]
    if data_type not in ["twitter.pkl", "reddit.pkl", "citation.pkl"]:
        auc = roc_auc_score(ground_truths, preds, average="micro")
        results["auc"] = auc
    else:
        accuracy = accuracy_score(ground_truths, y_label)
        results["accuracy"] = accuracy
    
    return results


@torch.no_grad()
def eval_link_prediction(model, link_predictor, dataloaders, evaluator, device):
    model.eval()
    link_predictor.eval()
    
    train_loader = dataloaders["train"]
    val_loader = dataloaders["valid"]
    val_negative_loader = dataloaders["valid_negative"]
    test_loader = dataloaders["test"]
    test_negative_loader = dataloaders["test_negative"]
    
    def test_split(dataloader, neg_dataloader, device):
        pos_preds = []
        for batch in dataloader:
            batch = batch.to(device)
            h = model(batch.x, batch.adj_t)
            src = batch.edge_label_index.t()[:, 0]
            dst = batch.edge_label_index.t()[:, 1]
            pos_preds += [link_predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)
        
        neg_preds = []
        for batch in neg_dataloader:
            batch = batch.to(device)
            h = model(batch.x, batch.adj_t)
            src = batch.edge_label_index.t()[:, 0]
            dst = batch.edge_label_index.t()[:, 1]
            neg_preds += [link_predictor(h[src], h[dst]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0)
        
        return pos_pred, neg_pred
    
    pos_train_pred, neg_train_pred = test_split(train_loader, val_negative_loader, device)
    pos_valid_pred, neg_valid_pred = test_split(val_loader, val_negative_loader, device)
    pos_test_pred, neg_test_pred = test_split(test_loader, test_negative_loader, device)
    
    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            "y_pred_pos": pos_train_pred,
            "y_pred_neg": neg_valid_pred,
        })[f"hits@{K}"]
        
        valid_hits = evaluator.eval({
            "y_pred_pos": pos_valid_pred,
            "y_pred_neg": neg_valid_pred,
        })[f"hits@{K}"]
        
        test_hits = evaluator.eval({
            "y_pred_pos": pos_test_pred,
            "y_pred_neg": neg_test_pred,
        })[f"hits@{K}"]
        
        results[f"Hits@{K}"] = (train_hits, valid_hits, test_hits)
    
    return results


# 新增：评估子图分类
@torch.no_grad()
def eval_subgraph_classification(model, subgraph_classifier, subgraph_loader, device):
    model.eval()
    subgraph_classifier.eval()
    
    # 预先计算所有节点的嵌入
    data = Data(x=subgraph_loader.data.x, adj_t=subgraph_loader.data.adj_t)
    h_all = model(data.x.to(device), data.adj_t.to(device))
    
    all_preds = []
    all_labels = []
    
    for batch_subgraphs, batch_labels in subgraph_loader:
        # 计算每个子图的嵌入
        subgraph_embeddings = []
        for subgraph in batch_subgraphs:
            subgraph = subgraph.to(device)
            subgraph_node_embeddings = h_all[subgraph]
            subgraph_embedding = torch.mean(subgraph_node_embeddings, dim=0)
            subgraph_embeddings.append(subgraph_embedding)
        
        subgraph_embeddings = torch.stack(subgraph_embeddings)
        
        # 子图分类
        subgraph_preds = subgraph_classifier(subgraph_embeddings)
        subgraph_preds = torch.argmax(subgraph_preds, dim=1)
        
        all_preds.append(subgraph_preds.cpu())
        all_labels.append(batch_labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    accuracy = torch.sum(all_preds == all_labels).item() / len(all_labels)
    
    return {"accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Multi-Task Learning for GNNs")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gnn_model", type=str, default="GAT", 
                        help="GNN Model: GAT, GraphTransformer, GINE, GeneralGNN")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--neg_len", type=str, default="5000")
    parser.add_argument("--node_weight", type=float, default=1.0,
                        help="Weight for node classification loss")
    parser.add_argument("--link_weight", type=float, default=1.0,
                        help="Weight for link prediction loss")
    parser.add_argument("--subgraph_weight", type=float, default=1.0,
                        help="Weight for subgraph classification loss")
    parser.add_argument("--min_subgraph_size", type=int, default=5,
                        help="Minimum size of subgraphs to use")
    parser.add_argument("--use_PLM_node", type=str,
                        default="data/CSTAG/Photo/Feature/children_gpt_node.pt",
                        help="Use LM embedding as node feature")
    parser.add_argument("--use_PLM_edge", type=str,
                        default="data/CSTAG/Photo/Feature/children_gpt_edge.pt",
                        help="Use LM embedding as edge feature")
    parser.add_argument("--path", type=str,
                        default="data/CSTAG/Photo/LinkPrediction/",
                        help="Path to save splitting")
    parser.add_argument("--graph_path", type=str,
                        default="data/CSTAG/Photo/children.pkl",
                        help="Path to load the graph")
    
    args = parser.parse_args()
    print(args)
    
    # 设置wandb
    wandb.init(config=args, project="multitask-gnn", reinit=True)
    
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
    class LinkEvaluator(object):
        def __init__(self, K=100):
            self.K = K

        def eval(self, input_dict):
            y_pred_pos, y_pred_neg = input_dict["y_pred_pos"], input_dict["y_pred_neg"]
            
            # Calculate Hits@K
            hits_k = self.calculate_hits_at_k(y_pred_pos, y_pred_neg)
            return {f"hits@{self.K}": hits_k}
            
        def calculate_hits_at_k(self, pos_pred, neg_pred):
            # Concatenate positive and negative predictions
            scores = torch.cat([pos_pred, neg_pred])
            labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
            
            # Get indices that would sort scores in descending order
            _, indices = torch.sort(scores, descending=True)
            
            # Sort labels according to scores
            sorted_labels = labels[indices]
            
            # Calculate hits (true positives) within top K predictions
            hits_k = torch.sum(sorted_labels[:self.K] == 1).item() / min(self.K, len(pos_pred))
            return hits_k
    
    # 初始化评估器
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
        
        # 设置优化器
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
        node_classifier.load_state_dict(torch.load(f"{args.path}/best_node_classifier.pt"))
        link_predictor.load_state_dict(torch.load(f"{args.path}/best_link_predictor.pt"))
        subgraph_classifier.load_state_dict(torch.load(f"{args.path}/best_subgraph_classifier.pt"))
        
        # 节点分类测试
        test_node_results = eval_node_classification(
            model, node_classifier, node_test_loader, args, device
        )
        
        # 链接预测测试
        test_link_results = eval_link_prediction(
            model, link_predictor, link_dataloaders, link_evaluator, device
        )
        
        # 子图分类测试
        test_subgraph_results = eval_subgraph_classification(
            model, subgraph_classifier, subgraph_test_loader, device
        )
        
        # 打印和记录测试结果
        print("\nTest Results:")
        print(f"Node Classification F1: {test_node_results['f1']:.4f}")
        if 'auc' in test_node_results:
            print(f"Node Classification AUC: {test_node_results['auc']:.4f}")
        if 'accuracy' in test_node_results:
            print(f"Node Classification Accuracy: {test_node_results['accuracy']:.4f}")
        
        print(f"Link Prediction Hits@10: {test_link_results['Hits@10'][2]:.4f}")
        print(f"Link Prediction Hits@50: {test_link_results['Hits@50'][2]:.4f}")
        print(f"Link Prediction Hits@100: {test_link_results['Hits@100'][2]:.4f}")
        
        print(f"Subgraph Classification Accuracy: {test_subgraph_results['accuracy']:.4f}")
        
        wandb.log({
            "Test/Node_F1": test_node_results['f1'],
            "Test/Link_Hits@10": test_link_results['Hits@10'][2],
            "Test/Link_Hits@50": test_link_results['Hits@50'][2],
            "Test/Link_Hits@100": test_link_results['Hits@100'][2],
            "Test/Subgraph_Accuracy": test_subgraph_results['accuracy']
        })
        
        if 'auc' in test_node_results:
            wandb.log({"Test/Node_AUC": test_node_results['auc']})
        if 'accuracy' in test_node_results:
            wandb.log({"Test/Node_Accuracy": test_node_results['accuracy']})
    
    # 关闭wandb
    wandb.finish()


if __name__ == "__main__":
    main()
