from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
import argparse
import pickle
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from model.GNN_library import GAT, GINE, GeneralGNN, GraphTransformer
from model.Dataloader import Evaluator, split_edge_mrr
from model.GNN_arg import Logger
import wandb
import os
import tqdm

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
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

def train(model, predictor, loader, optimizer, device):
    model.train()
    predictor.train()
    total_loss = 0
    for batch in tqdm.tqdm(loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        h = model(batch.x, batch.adj_t)
        
        src, dst = batch.edge_label_index.t().chunk(2, dim=1)
        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        
        dst_neg = torch.randint(0, h.size(0), src.size(), device=device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * src.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(model, predictor, loader, evaluator, neg_len, device):
    model.eval()
    predictor.eval()
    pos_pred, neg_pred = [], []
    
    for batch in tqdm.tqdm(loader):
        batch = batch.to(device)
        h = model(batch.x, batch.adj_t)
        
        src, dst = batch.edge_label_index.t().chunk(2, dim=1)
        pos_pred.append(predictor(h[src], h[dst]).squeeze().cpu())
        
        dst_neg = torch.randint(0, h.size(0), (src.size(0), neg_len), device=device)
        src = src.repeat_interleave(neg_len)
        neg_pred.append(predictor(h[src], h[dst_neg.view(-1)]).view(-1, neg_len).cpu())
    
    pos_pred = torch.cat(pos_pred)
    neg_pred = torch.cat(neg_pred)
    return evaluator.eval({"y_pred_pos": pos_pred, "y_pred_neg": neg_pred})["mrr_list"].mean().item()

def gen_embeddings(model, data, device, batch_size=32768):
    model.eval()
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[20, 10],
        edge_label_index=torch.zeros((2, 0), dtype=torch.long),
        batch_size=batch_size,
        shuffle=False
    )
    
    embeddings = []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            embeddings.append(model(batch.x.to(device), batch.adj_t.to(device)).cpu())
    return torch.cat(embeddings)

def main():
    parser = argparse.ArgumentParser(description="Link Prediction with Optimal Model Saving")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hidden_channels", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gnn_model", type=str, default="GraphTransformer")
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--use_PLM_node", type=str, required=True)
    parser.add_argument("--use_PLM_edge", type=str, required=True)
    parser.add_argument("--graph_path", type=str, required=True)
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    wandb.init(config=args)

    # 数据加载
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    with open(args.graph_path, "rb") as f:
        graph = pickle.load(f)
    
    # 数据处理
    edge_split = split_edge_mrr(graph, args.test_ratio, args.val_ratio)
    x = torch.load(args.use_PLM_node).float()
    edge_feature = torch.load(args.use_PLM_edge)[edge_split["train"]["train_edge_feature_index"]].float()
    
    adj_t = SparseTensor.from_edge_index(
        edge_split["train"]["edge"].t(),
        edge_feature,
        sparse_sizes=(graph.num_nodes, graph.num_nodes)
    ).to_symmetric()
    
    # 数据加载器
    train_loader = LinkNeighborLoader(
        Data(x=x, adj_t=adj_t),
        num_neighbors=[20, 10],
        edge_label_index=edge_split["train"]["edge"].t(),
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # 全局最优模型跟踪
    best_state = {"mrr": 0.0, "model": None, "predictor": None}
    
    for run in range(args.runs):
        # 模型初始化
        model = GAT(x.size(1), edge_feature.size(1), args.hidden_channels, 
                   args.hidden_channels, 3, 4, args.dropout).to(device)
        predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1, 3, args.dropout).to(device)
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': predictor.parameters()}
        ], lr=args.lr)
        
        # 当前run的最佳模型
        run_best = {"mrr": 0.0, "state": None}
        
        for epoch in range(1, args.epochs + 1):
            loss = train(model, predictor, train_loader, optimizer, device)
            
            if epoch % args.eval_steps == 0:
                val_mrr = test(model, predictor, 
                              LinkNeighborLoader(
                                  Data(x=x, adj_t=adj_t),
                                  edge_label_index=edge_split["valid"]["edge"].t(),
                                  batch_size=args.batch_size,
                                  shuffle=False
                              ),
                              Evaluator(name="Val"), args.neg_len, device)
                
                # 更新run最佳
                if val_mrr > run_best["mrr"]:
                    run_best.update(mrr=val_mrr, state={
                        "model": model.state_dict(),
                        "predictor": predictor.state_dict()
                    })
                    
                # 更新全局最佳
                if val_mrr > best_state["mrr"]:
                    best_state.update(mrr=val_mrr, model=model.state_dict(), 
                                    predictor=predictor.state_dict())
                    torch.save(best_state, f"{args.save_dir}/best_model.pth")

        # 释放当前run资源
        del model, predictor, optimizer
        
    # 生成最终嵌入
    print(f"\nBest Validation MRR: {best_state['mrr']:.4f}")
    final_model = GAT(x.size(1), edge_feature.size(1), args.hidden_channels, 
                     args.hidden_channels, 3, 4, args.dropout).to(device)
    final_model.load_state_dict(best_state["model"])
    
    embeddings = gen_embeddings(
        final_model, 
        Data(x=x, adj_t=adj_t), 
        device,
        batch_size=args.batch_size
    )
    torch.save(embeddings, f"{args.save_dir}/final_embeddings.pt")
    print(f"Embeddings saved with shape {embeddings.shape}")

if __name__ == "__main__":
    main()
