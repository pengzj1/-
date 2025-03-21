import torch
import pickle
import tqdm
import argparse
import os
import wandb

from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from torch.nn import Linear

from model.GNN_library import GAT, GINE, GeneralGNN, GraphTransformer
from model.Dataloader import Evaluator, split_edge


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


def train_multitask(model, node_classifier, link_predictor, node_loader, link_loader, optimizer, node_criterion, device, args):
    model.train()
    node_classifier.train()
    link_predictor.train()

    # 定义损失权重
    node_weight = args.node_weight
    link_weight = args.link_weight

    total_loss = 0
    node_loss_sum = 0
    link_loss_sum = 0
    
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
    
    avg_node_loss = node_loss_sum / len(node_loader)
    avg_link_loss = link_loss_sum / len(link_loader)
    
    return {
        "total_loss": total_loss / (len(node_loader) + len(link_loader)),
        "node_loss": avg_node_loss,
        "link_loss": avg_link_loss
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
    
    # 创建模型
    model = gen_model(args, x, edge_feature).to(device)
    node_classifier = Classifier(args.hidden_channels, y.shape[1]).to(device)
    link_predictor = LinkPredictor(
        args.hidden_channels, args.hidden_channels, 1, args.num_layers, args.dropout
    ).to(device)
    
    # 创建评估器
    evaluator = Evaluator(name="History")
    node_criterion = torch.nn.BCEWithLogitsLoss()
    
    # 运行多次实验
    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}")
        
        # 重置模型参数
        if hasattr(model, 'reset_parameters'):
            model.reset_parameters()
        if hasattr(node_classifier, 'reset_parameters'):
            node_classifier.reset_parameters()
        link_predictor.reset_parameters()
        
        # 创建优化器
        optimizer = torch.optim.Adam(
            list(model.parameters()) + 
            list(node_classifier.parameters()) + 
            list(link_predictor.parameters()), 
            lr=args.lr
        )
        
        # 训练与评估
        best_val_f1 = 0
        best_val_hits = 0
        best_epoch = 0
        
        for epoch in range(1, 1 + args.epochs):
            # 训练
            loss_dict = train_multitask(
                model, node_classifier, link_predictor, 
                node_train_loader, link_train_loader, 
                optimizer, node_criterion, device, args
            )
            
            print(f"Epoch: {epoch:03d}, "
                  f"Total Loss: {loss_dict['total_loss']:.4f}, "
                  f"Node Loss: {loss_dict['node_loss']:.4f}, "
                  f"Link Loss: {loss_dict['link_loss']:.4f}")
            
            # 记录到wandb
            wandb.log({
                "epoch": epoch,
                "total_loss": loss_dict["total_loss"],
                "node_loss": loss_dict["node_loss"],
                "link_loss": loss_dict["link_loss"]
            })
            
            # 评估
            if epoch % args.eval_steps == 0:
                # 节点分类评估
                node_results = eval_node_classification(
                    model, node_classifier, node_val_loader, args, device
                )
                
                # 链接预测评估
                link_results = eval_link_prediction(
                    model, link_predictor, link_dataloaders, evaluator, device
                )
                
                # 打印结果
                print("Validation Results:")
                print(f"  Node Classification F1: {node_results['f1']:.4f}")
                if "auc" in node_results:
                    print(f"  Node Classification AUC: {node_results['auc']:.4f}")
                if "accuracy" in node_results:
                    print(f"  Node Classification Accuracy: {node_results['accuracy']:.4f}")
                
                for k, result in link_results.items():
                    train_hits, valid_hits, _ = result
                    print(f"  Link Prediction {k}: Train: {100 * train_hits:.2f}%, Valid: {100 * valid_hits:.2f}%")
                
                # 记录到wandb
                log_dict = {
                    "val_f1": node_results["f1"],
                    "val_hits@10": link_results["Hits@10"][1],
                    "val_hits@50": link_results["Hits@50"][1],
                    "val_hits@100": link_results["Hits@100"][1]
                }
                
                if "auc" in node_results:
                    log_dict["val_auc"] = node_results["auc"]
                if "accuracy" in node_results:
                    log_dict["val_accuracy"] = node_results["accuracy"]
                
                wandb.log(log_dict)
                
                # 保存最佳模型
                current_val_f1 = node_results["f1"]
                current_val_hits = link_results["Hits@10"][1]
                
                if current_val_f1 > best_val_f1 or current_val_hits > best_val_hits:
                    best_val_f1 = max(current_val_f1, best_val_f1)
                    best_val_hits = max(current_val_hits, best_val_hits)
                    best_epoch = epoch
                    
                    # 保存模型
                    torch.save(model.state_dict(), f"model_run{run}_best.pt")
                    torch.save(node_classifier.state_dict(), f"node_classifier_run{run}_best.pt")
                    torch.save(link_predictor.state_dict(), f"link_predictor_run{run}_best.pt")
        
        # 加载最佳模型
        model.load_state_dict(torch.load(f"model_run{run}_best.pt"))
        node_classifier.load_state_dict(torch.load(f"node_classifier_run{run}_best.pt"))
        link_predictor.load_state_dict(torch.load(f"link_predictor_run{run}_best.pt"))
        
        # 在测试集上评估
        print(f"\nTest Results (Best model from epoch {best_epoch}):")
        
        # 节点分类评估
        node_test_results = eval_node_classification(
            model, node_classifier, node_test_loader, args, device
        )
        
        # 链接预测评估
        link_test_results = eval_link_prediction(
            model, link_predictor, link_dataloaders, evaluator, device
        )
        
        # 打印结果
        print(f"  Node Classification F1: {node_test_results['f1']:.4f}")
        if "auc" in node_test_results:
            print(f"  Node Classification AUC: {node_test_results['auc']:.4f}")
        if "accuracy" in node_test_results:
            print(f"  Node Classification Accuracy: {node_test_results['accuracy']:.4f}")
        
        for k, result in link_test_results.items():
            _, _, test_hits = result
            print(f"  Link Prediction {k}: Test: {100 * test_hits:.2f}%")
        
        # 记录到wandb
        test_log_dict = {
            "test_f1": node_test_results["f1"],
            "test_hits@10": link_test_results["Hits@10"][2],
            "test_hits@50": link_test_results["Hits@50"][2],
            "test_hits@100": link_test_results["Hits@100"][2]
        }
        
        if "auc" in node_test_results:
            test_log_dict["test_auc"] = node_test_results["auc"]
        if "accuracy" in node_test_results:
            test_log_dict["test_accuracy"] = node_test_results["accuracy"]
        
        wandb.log(test_log_dict)
    
    wandb.finish()


if __name__ == "__main__":
    seed_everything(66)
    main()