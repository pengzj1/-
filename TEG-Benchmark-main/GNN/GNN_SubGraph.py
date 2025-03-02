import torch
import pickle
import numpy as np
import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from model.GNN_library import GAT, GINE, GeneralGNN, GraphTransformer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import argparse
from networkx.algorithms.community import greedy_modularity_communities
from torch_geometric.utils import to_networkx

class WholeGraphProcessor:
    def __init__(self, args):
        # 加载原始大图数据
        with open(args.graph_path, "rb") as f:
            self.raw_data = pickle.load(f)
        
        # 加载PLM特征
        self.node_features = torch.load(args.use_PLM_node).squeeze().float()
        self.edge_features = torch.load(args.use_PLM_edge).squeeze().float()
        
        # 构建PyG数据对象
        self.data = Data(
            x=self.node_features,
            edge_index=self.raw_data.edge_index,
            edge_attr=self.edge_features,
            y=self._process_labels()
        )
        
    def _process_labels(self):
        # 保持与原节点分类相同的标签处理逻辑
        product_indices = torch.tensor([
            i for i, label in enumerate(self.raw_data.text_node_labels) if label != -1
        ])
        mlb = MultiLabelBinarizer()
        product_binary_labels = mlb.fit_transform(self.raw_data.text_node_labels[product_indices])
        y = torch.zeros(self.raw_data.num_nodes, product_binary_labels.shape[1]).float()
        y[product_indices] = torch.tensor(product_binary_labels).float()
        return y

class SubgraphDataset(Dataset):
    def __init__(self, whole_graph_data, num_subgraphs=1000, min_size=20):
        super().__init__()
        self.whole_data = whole_graph_data
        self.subgraphs = []
        self._generate_subgraphs(num_subgraphs, min_size)
        
    def _generate_subgraphs(self, num_subgraphs, min_size):
        # 转换为NetworkX图进行社区发现
        G = to_networkx(self.whole_data, node_attrs=['y'], edge_attrs=['edge_attr'])
        communities = list(greedy_modularity_communities(G))
        
        for comm in communities[:num_subgraphs]:
            if len(comm) < min_size:
                continue
                
            # 创建子图数据
            sub_nodes = sorted(comm)
            sub_data = self._create_subgraph(sub_nodes)
            self.subgraphs.append(sub_data)
            
    def _create_subgraph(self, sub_nodes):
        # 节点掩码
        node_mask = torch.zeros(self.whole_data.num_nodes, dtype=torch.bool)
        node_mask[sub_nodes] = True
        
        # 提取子图边
        row, col = self.whole_data.edge_index
        edge_mask = node_mask[row] & node_mask[col]
        
        # 子图标签（多标签分类）
        sub_labels = self.whole_data.y[sub_nodes]
        graph_label = (sub_labels.sum(dim=0) > 0).float()  # 存在即标记
        
        return Data(
            x=self.whole_data.x[sub_nodes],
            edge_index=self.whole_data.edge_index[:, edge_mask],
            edge_attr=self.whole_data.edge_attr[edge_mask],
            y=graph_label.unsqueeze(0),
            sub_node_indices=torch.tensor(sub_nodes)
        )

    def len(self):
        return len(self.subgraphs)

    def get(self, idx):
        return self.subgraphs[idx]

class SubgraphGNN(torch.nn.Module):
    def __init__(self, gnn, hidden_dim, num_classes):
        super().__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, data):
        # 处理边特征
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # 节点嵌入
        h = self.gnn(data.x, data.edge_index, edge_attr)
        
        # 图池化
        graph_emb = self.pool(h, torch.zeros(h.size(0), dtype=torch.long, device=h.device))
        
        # 分类
        return self.classifier(graph_emb)

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch).squeeze()
        loss = criterion(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        
    return total_loss / len(loader.dataset)

def evaluate(model, loader, threshold=0.5):
    model.eval()
    preds, truths = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)
            out = model(batch).sigmoid().squeeze()
            preds.append(out.cpu())
            truths.append(batch.y.squeeze().cpu())
    
    preds = torch.cat(preds).numpy()
    truths = torch.cat(truths).numpy()
    
    # 多标签评估指标
    y_pred = (preds > threshold).astype(int)
    return {
        'accuracy': accuracy_score(truths, y_pred),
        'f1': f1_score(truths, y_pred, average='macro'),
        'auc': roc_auc_score(truths, preds, average='macro')
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 保持原始参数结构
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gnn_model", type=str, default="GAT")
    parser.add_argument("--use_PLM_node", type=str, required=True)
    parser.add_argument("--use_PLM_edge", type=str, required=True)
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_subgraphs", type=int, default=1000)
    parser.add_argument("--min_subgraph_size", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # 处理原始大图
    processor = WholeGraphProcessor(args)
    
    # 创建子图数据集
    dataset = SubgraphDataset(
        processor.data,
        num_subgraphs=args.num_subgraphs,
        min_size=args.min_subgraph_size
    )
    
    # 数据集划分
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size+val_size]
    test_dataset = dataset[train_size+val_size:]
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 初始化模型
    base_gnn = GAT(
        processor.data.x.size(1),
        processor.data.edge_attr.size(1),
        args.hidden_channels,
        args.hidden_channels,
        args.num_layers,
        heads=4,
        dropout=args.dropout,
    )
    model = SubgraphGNN(
        base_gnn,
        args.hidden_channels,
        num_classes=processor.data.y.size(1)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 训练循环
    best_auc = 0
    for epoch in range(1, args.epochs+1):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader)
        
        print(f"Epoch {epoch:03d}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val AUC: {val_metrics['auc']:.4f} | F1: {val_metrics['f1']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
        
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(model.state_dict(), "best_model.pth")
    
    # 最终测试
    model.load_state_dict(torch.load("best_model.pth"))
    test_metrics = evaluate(model, test_loader)
    print("\nFinal Test Results:")
    print(f"AUC: {test_metrics['auc']:.4f} | F1: {test_metrics['f1']:.4f} | Accuracy: {test_metrics['accuracy']:.4f}")
