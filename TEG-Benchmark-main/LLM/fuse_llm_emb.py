import argparse
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class EmbeddingFuser:
    def __init__(self, strategy='concat'):
        self.strategy = strategy
        self.bert_scaler = StandardScaler()
        self.llama_scaler = StandardScaler()

    def _load_tensor(self, path):
        """加载并验证pt文件"""
        emb = torch.load(path)
        if not isinstance(emb, torch.Tensor):
            raise ValueError(f"非张量格式: {type(emb)} in {path}")
        return emb.numpy()  # 转换为numpy用于后续处理

    def load_embeddings(self, bert_path, llama_path):
        """加载并预处理嵌入"""
        bert_emb = self._load_tensor(bert_path)
        llama_emb = self._load_tensor(llama_path)

        # 维度验证
        if bert_emb.shape[0] != llama_emb.shape[0]:
            raise ValueError(f"样本数不匹配: BERT({bert_emb.shape[0]}) vs LLaMA({llama_emb.shape[0]})")

        # 标准化
        return (
            self.bert_scaler.fit_transform(bert_emb),
            self.llama_scaler.fit_transform(llama_emb)
        )

    def fuse(self, bert_emb, llama_emb):
        """执行融合操作"""
        if self.strategy == 'concat':
            fused = np.concatenate([bert_emb, llama_emb], axis=1)
        
        elif self.strategy == 'weighted_sum':
            if bert_emb.shape[1] != llama_emb.shape[1]:
                raise ValueError("加权求和需要相同维度")
            fused = 0.6 * bert_emb + 0.4 * llama_emb
        
        elif self.strategy == 'pca_concat':
            pca = PCA(n_components=min(256, bert_emb.shape[1]))
            fused = np.concatenate([
                pca.fit_transform(bert_emb),
                pca.fit_transform(llama_emb)
            ], axis=1)
        
        else:
            raise ValueError(f"未知策略: {self.strategy}")

        # 转换为PyTorch张量并优化存储
        return torch.from_numpy(fused).to(torch.float16)  # 半精度存储

def main():
    parser = argparse.ArgumentParser(description='多模态嵌入融合工具')
    parser.add_argument('--bert', required=True, help='BERT嵌入路径(.pt)')
    parser.add_argument('--llama', required=True, help='LLaMA嵌入路径(.pt)')
    parser.add_argument('--output', default='fused.pt', help='输出文件路径')
    parser.add_argument('--strategy', choices=['concat', 'weighted_sum', 'pca_concat'], 
                      default='concat', help='融合策略')
    
    args = parser.parse_args()
    
    try:
        # 初始化融合器
        fuser = EmbeddingFuser(args.strategy)
        
        # 加载和预处理
        print("正在加载和预处理嵌入...")
        bert_emb, llama_emb = fuser.load_embeddings(args.bert, args.llama)
        
        # 执行融合
        print(f"执行 {args.strategy} 融合...")
        fused_tensor = fuser.fuse(bert_emb, llama_emb)
        
        # 保存结果
        torch.save(fused_tensor, args.output)
        print(f"融合完成! 保存至: {args.output}")
        print(f"输出维度: {fused_tensor.shape[1]} (形状: {fused_tensor.shape})")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()


# 基础用法
python fuse_embeddings.py \
  --bert bert_embeddings.pt \
  --llama llama_embeddings.pt \
  --output fused_emb.pt \
  --strategy pca_concat

# 验证输出文件
python -c "import torch; emb = torch.load('fused_emb.pt'); print(f'Shape: {emb.shape} | Dtype: {emb.dtype}')"
