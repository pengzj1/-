import torch
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

from torch.utils.data.sampler import Sampler

class WeightedRandomSampler(Sampler):
    """加权随机采样器：处理类别不平衡问题"""
    def __init__(self, labels, num_samples=None, replacement=True):
        self.labels = labels
        self.replacement = replacement
        self.weights = self._compute_weights(labels)
        self.num_samples = num_samples if num_samples is not None else len(labels)
    
    def _compute_weights(self, labels):
        """计算每个样本的权重（类频率的倒数）"""
        label_counts = torch.bincount(labels)
        class_weights = 1.0 / label_counts.float()
        # 处理可能的零除
        class_weights[torch.isinf(class_weights)] = 0
        
        sample_weights = class_weights[labels]
        return sample_weights
    
    def __iter__(self):
        """返回加权采样的索引迭代器"""
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())
    
    def __len__(self):
        """返回采样数量"""
        return self.num_samples


def create_data_loaders(graph_embeds, text_embeds, labels, batch_size=32):
    """创建训练、验证和测试数据加载器"""

    # 确保所有张量都是标准精度浮点型
    graph_embeds = graph_embeds.to(torch.float32)
    text_embeds = text_embeds.to(torch.float32)
    
    # 创建数据集
    dataset = TensorDataset(graph_embeds, text_embeds, labels)
    
    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建加权采样器处理类别不平衡
    train_labels = torch.tensor([dataset[i][2] for i in train_dataset.indices])
    sampler = WeightedRandomSampler(train_labels, num_samples=len(train_dataset))

    # 创建数据加载器
    train_loader = DataLoader(
    train_dataset, batch_size=batch_size, sampler=sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader