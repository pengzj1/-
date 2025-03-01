import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
import argparse
import pickle
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from model.GNN_library import GAT, GINE, GeneralGNN, GraphTransformer
from model.Dataloader import Evaluator, split_edge
from model.GNN_arg import Logger
import wandb
import os

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
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

def train(model, predictor, train_loader, optimizer, device):
    model.train()
    predictor.train()
    total_loss = total_examples = 0
    for batch in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        h = model(batch.x, batch.adj_t)

        src = batch.edge_label_index.t()[:, 0]
        dst = batch.edge_label_index.t()[:, 1]

        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        dst_neg = torch.randint(0, h.size(0), src.size(), dtype=torch.long, device=h.device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * pos_out.size(0)
        total_examples += pos_out.size(0)
    return total_loss / total_examples

@torch.no_grad()
def test(model, predictor, dataloaders, evaluator, device):
    model.eval()
    predictor.eval()
    (train_loader, valid_loader, valid_negative_loader, 
     test_loader, test_negative_loader) = (dataloaders[k] for k in 
        ["train", "valid", "valid_negative", "test", "test_negative"])

    def test_split(dataloader, neg_dataloader):
        pos_preds, neg_preds = [], []
        for batch in dataloader:
            batch = batch.to(device)
            h = model(batch.x, batch.adj_t)
            src, dst = batch.edge_label_index.t().chunk(2, dim=1)
            pos_preds.append(predictor(h[src], h[dst]).squeeze().cpu())
        for batch in neg_dataloader:
            batch = batch.to(device)
            h = model(batch.x, batch.adj_t)
            src, dst = batch.edge_label_index.t().chunk(2, dim=1)
            neg_preds.append(predictor(h[src], h[dst]).squeeze().cpu())
        return torch.cat(pos_preds), torch.cat(neg_preds)

    pos_train, neg_valid = test_split(train_loader, valid_negative_loader)
    pos_valid, _ = test_split(valid_loader, valid_negative_loader)
    pos_test, neg_test = test_split(test_loader, test_negative_loader)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({"y_pred_pos": pos_train, "y_pred_neg": neg_valid})[f"hits@{K}"]
        valid_hits = evaluator.eval({"y_pred_pos": pos_valid, "y_pred_neg": neg_valid})[f"hits@{K}"]
        test_hits = evaluator.eval({"y_pred_pos": pos_test, "y_pred_neg": neg_test})[f"hits@{K}"]
        results[f"Hits@{K}"] = (train_hits, valid_hits, test_hits)
    return results

def gen_model(args, x, edge_feature):
    gnn_classes = {
        "GAT": GAT,
        "GraphTransformer": GraphTransformer,
        "GINE": GINE,
        "GeneralGNN": GeneralGNN
    }
    return gnn_classes[args.gnn_model](
        x.size(1), edge_feature.size(1), args.hidden_channels, 
        args.hidden_channels, args.num_layers, getattr(args, 'heads', 4), args.dropout
    )

def gen_loader(args, edge_split, x, edge_index, adj_t):
    def create_loader(edge_index, **kwargs):
        return LinkNeighborLoader(
            Data(x=x, adj_t=adj_t),
            num_neighbors=[20, 10],
            edge_label_index=edge_index.t(),
            edge_label=torch.ones(edge_index.size(1)),
            batch_size=args.batch_size,
            **kwargs
        )
    return (
        create_loader(edge_split["train"]["edge"], shuffle=True),
        create_loader(edge_split["valid"]["edge"], shuffle=False),
        create_loader(edge_split["valid"]["edge_neg"], shuffle=False),
        create_loader(edge_split["test"]["edge"], shuffle=False),
        create_loader(edge_split["test"]["edge_neg"], shuffle=False),
    )

@torch.no_grad()
def get_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    for batch in tqdm.tqdm(loader):
        batch = batch.to(device)
        embeddings.append(model(batch.x, batch.adj_t).cpu())
    return torch.cat(embeddings)

def main():
    parser = argparse.ArgumentParser(description="Link Prediction with Graph Embeddings")
    # 参数设置
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gnn_model", type=str, default="GraphTransformer")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    # 数据路径
    parser.add_argument("--use_PLM_node", type=str, required=True)
    parser.add_argument("--use_PLM_edge", type=str, required=True)
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--path", type=str, default="./data/")
    
    args = parser.parse_args()
    wandb.init(config=args)
    os.makedirs(args.save_dir, exist_ok=True)

    # 数据加载与预处理
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    with open(args.graph_path, "rb") as f:
        graph = pickle.load(f)
    edge_split = split_edge(graph, args.test_ratio, args.val_ratio)
    
    # 特征处理
    x = torch.load(args.use_PLM_node).float()
    edge_feature = torch.load(args.use_PLM_edge)[edge_split["train"]["train_edge_feature_index"]].float()
    adj_t = SparseTensor.from_edge_index(
        edge_split["train"]["edge"].t(), edge_feature, 
        sparse_sizes=(graph.num_nodes, graph.num_nodes)
    ).to_symmetric()

    # 数据加载器
    dataloaders = dict(zip(
        ["train", "valid", "valid_negative", "test", "test_negative"],
        gen_loader(args, edge_split, x, edge_split["train"]["edge"], adj_t)
    ))

    # 模型初始化
    model = gen_model(args, x, edge_feature).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1, 3, args.dropout).to(device)
    evaluator = Evaluator(name="LinkPrediction")
    loggers = {f"Hits@{K}": Logger(args.runs, args) for K in [10, 50, 100]}

    best_metrics = {"Hits@100": 0}
    best_model = None

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()), lr=args.lr
        )

        for epoch in range(1, args.epochs + 1):
            loss = train(model, predictor, dataloaders["train"], optimizer, device)
            
            if epoch % args.eval_steps == 0:
                results = test(model, predictor, dataloaders, evaluator, device)
                current_metric = results["Hits@100"][1]  # 使用验证集Hits@100
                
                # 保存最佳模型
                if current_metric > best_metrics["Hits@100"]:
                    best_metrics = results
                    best_model = {
                        "model": model.state_dict(),
                        "predictor": predictor.state_dict(),
                        "epoch": epoch
                    }
                    torch.save(best_model, f"{args.save_dir}/best_model.pth")

                # 日志记录
                wandb.log({
                    "epoch": epoch,
                    "train_loss": loss,
                    **{f"val_{k}": v[1] for k, v in results.items()},
                    **{f"test_{k}": v[2] for k, v in results.items()}
                })

        # 生成嵌入
        print("\nGenerating Graph Embeddings...")
        # 加载最佳模型
        checkpoint = torch.load(f"{args.save_dir}/best_model.pth")
        model.load_state_dict(checkpoint["model"])
        
        # 创建全图数据加载器
        full_loader = LinkNeighborLoader(
            Data(x=x, adj_t=adj_t),
            num_neighbors=[20, 10],
            edge_label_index=torch.zeros((2, 0), dtype=torch.long),  # 空边标签
            batch_size=args.batch_size,
            shuffle=False
        )
        
        # 获取所有节点嵌入
        embeddings = get_embeddings(model, full_loader, device)
        
        # 保存嵌入结果
        torch.save(embeddings, f"{args.save_dir}/{args.gnn_model}_embeddings.pt")
        print(f"Saved {embeddings.size(0)} node embeddings to {args.save_dir}")

if __name__ == "__main__":
    main()


#   python main.py \
  --use_PLM_node path/to/node_features.pt \
  --use_PLM_edge path/to/edge_features.pt \
  --graph_path path/to/graph.pkl \
  --gnn_model GraphTransformer \
  --hidden_channels 512 \
  --batch_size 32768 \
  --save_dir ./embeddings
