import torch
import pickle
from torch_sparse import SparseTensor
import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric import seed_everything
from torch_geometric.loader import NeighborLoader
from model.GNN_library import GAT, GINE, GeneralGNN, GraphTransformer
from torch.nn import Linear
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import argparse
from torch_geometric.data import Data
from sklearn.preprocessing import MultiLabelBinarizer
import os

class Classifier(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x)
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

def train(model, predictor, train_loader, optimizer, criterion):
    model.train()
    predictor.train()
    total_loss = total_examples = 0
    for batch in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        h = model(batch.x, batch.adj_t)[:batch.batch_size]
        pred = predictor(h)
        loss = criterion(pred, batch.y[:batch.batch_size].float())
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    return total_loss / total_examples

def evaluate(model, predictor, loader, args):
    model.eval()
    predictor.eval()
    preds, ground_truths = [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            batch = batch.to(device)
            h = model(batch.x, batch.adj_t)[:batch.batch_size]
            pred = predictor(h)
            ground_truth = batch.y[:batch.batch_size]
            preds.append(pred)
            ground_truths.append(ground_truth)
    
    preds = torch.cat(preds, dim=0).cpu().numpy()
    ground_truths = torch.cat(ground_truths, dim=0).cpu().numpy()
    
    # 根据数据集类型选择评估指标
    data_type = os.path.basename(args.graph_path)
    if data_type not in ["twitter.pkl", "reddit.pkl", "citation.pkl"]:
        metric = roc_auc_score(ground_truths, preds, average='micro')
    else:
        y_label = (preds > args.threshold).astype(int)
        metric = accuracy_score(ground_truths, y_label)
    return metric

def get_embeddings(model, data_loader, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader):
            batch = batch.to(device)
            h = model(batch.x, batch.adj_t)[:batch.batch_size]
            embeddings.append(h.cpu())
    return torch.cat(embeddings, dim=0)

def process_data(args, device, data):
    num_nodes = len(data.text_nodes)
    product_indices = torch.tensor([
        i for i, label in enumerate(data.text_node_labels) if label != -1
    ]).long()
    product_labels = [label for label in data.text_node_labels if label != -1]
    
    mlb = MultiLabelBinarizer()
    product_binary_labels = mlb.fit_transform(product_labels)
    y = torch.zeros(num_nodes, product_binary_labels.shape[1]).float()
    y[product_indices] = torch.tensor(product_binary_labels).float()
    y = y.to(device)

    # 数据集划分
    train_ratio = 1 - args.test_ratio - args.val_ratio
    num_products = product_indices.shape[0]
    splits = {
        'train': product_indices[:int(num_products * train_ratio)],
        'val': product_indices[int(num_products * train_ratio):int(num_products * (train_ratio + args.val_ratio))],
        'test': product_indices[int(num_products * (train_ratio + args.val_ratio)):]
    }

    # 特征加载
    x = torch.load(args.use_PLM_node).squeeze().float()
    edge_feature = torch.load(args.use_PLM_edge).squeeze().float()
    
    # 构建邻接矩阵
    edge_index = data.edge_index
    adj_t = SparseTensor.from_edge_index(
        edge_index, 
        edge_attr=edge_feature,
        sparse_sizes=(num_nodes, num_nodes)
    ).t().to_symmetric()
    
    return splits, x, edge_feature, adj_t, y, product_indices

if __name__ == "__main__":
    seed_everything(66)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gnn_model", type=str, default="GAT")
    parser.add_argument("--use_PLM_node", type=str, required=True)
    parser.add_argument("--use_PLM_edge", type=str, required=True)
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)

    # 数据预处理
    with open(args.graph_path, "rb") as f:
        raw_data = pickle.load(f)
    splits, x, edge_feature, adj_t, y, product_indices = process_data(args, device, raw_data)
    data = Data(x=x, adj_t=adj_t, y=y)

    # 创建数据加载器
    loaders = {
        'train': NeighborLoader(data, input_nodes=splits['train'], 
                               num_neighbors=[10, 10], batch_size=args.batch_size, shuffle=True),
        'val': NeighborLoader(data, input_nodes=splits['val'], 
                             num_neighbors=[10, 10], batch_size=args.batch_size),
        'test': NeighborLoader(data, input_nodes=splits['test'], 
                              num_neighbors=[10, 10], batch_size=args.batch_size)
    }

    # 模型初始化
    model = gen_model(args, x, edge_feature).to(device)
    predictor = Classifier(args.hidden_channels, y.size(1)).to(device)
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': predictor.parameters()}
    ], lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 训练循环
    best_metric = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, predictor, loaders['train'], optimizer, criterion)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f}")

        if epoch % args.eval_steps == 0:
            val_metric = evaluate(model, predictor, loaders['val'], args)
            data_type = os.path.basename(args.graph_path)
            metric_name = "AUC" if data_type not in ["twitter.pkl", "reddit.pkl", "citation.pkl"] else "Accuracy"
            print(f"Validation {metric_name}: {val_metric:.4f}")

            # 保存最佳模型
            if val_metric > best_metric:
                best_metric = val_metric
                torch.save({
                    'model': model.state_dict(),
                    'predictor': predictor.state_dict(),
                    'epoch': epoch,
                    'args': vars(args)
                }, os.path.join(args.save_dir, "best_model.pth"))
                print(f"New best model saved at epoch {epoch} with {metric_name} {val_metric:.4f}")

    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.save_dir, "best_model.pth"))
    model.load_state_dict(checkpoint['model'])
    predictor.load_state_dict(checkpoint['predictor'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")

    # 最终测试
    test_metric = evaluate(model, predictor, loaders['test'], args)
    data_type = os.path.basename(args.graph_path)
    metric_name = "AUC" if data_type not in ["twitter.pkl", "reddit.pkl", "citation.pkl"] else "Accuracy"
    print(f"\nFinal Test {metric_name}: {test_metric:.4f}")

    # 生成嵌入
    print("\nGenerating embeddings...")
    embed_loader = NeighborLoader(
        data,
        input_nodes=torch.arange(data.num_nodes).to(device),
        num_neighbors=[10, 10],
        batch_size=args.batch_size,
        shuffle=False
    )
    embeddings = get_embeddings(model, embed_loader, device)
    
    # 保存嵌入
    save_path = os.path.join(args.save_dir, f"{args.gnn_model}_embeddings.pt")
    torch.save(embeddings, save_path)
    print(f"Saved {embeddings.size(0)} node embeddings to {save_path}")
