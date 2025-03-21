
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

def train_aligner_with_scheduler(model, train_loader, val_loader, num_epochs=50, 
                                 lr=0.001, weight_decay=1e-5, warmup_steps=100):
    """使用学习率调度、梯度裁剪和早停的训练函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 加入权重衰减正则化
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 温热学习率调度
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # 早停检测
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        all_losses = {}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # 将数据移到设备上
            batch = [item.to(device) for item in batch]
            
            optimizer.zero_grad()
            loss, losses, _ = model.compute_total_loss(batch)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # 更新进度条
            progress_bar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})
            
            epoch_train_loss += loss.item()
            
            # 累积各种损失值
            for k, v in losses.items():
                if k not in all_losses:
                    all_losses[k] = []
                all_losses[k].append(v)
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # 打印详细损失信息
        loss_str = " - ".join([f"{k}: {np.mean(v):.4f}" for k, v in all_losses.items()])
        print(f"Epoch {epoch+1} Train - Loss: {avg_train_loss:.4f} - {loss_str}")
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        val_losses = {}
        
        with torch.no_grad():
            for batch in val_loader:
                batch = [item.to(device) for item in batch]
                loss, losses, _ = model.compute_total_loss(batch)
                epoch_val_loss += loss.item()
                
                # 累积验证损失
                for k, v in losses.items():
                    if k not in val_losses:
                        val_losses[k] = []
                    val_losses[k].append(v)
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_loss_str = " - ".join([f"{k}: {np.mean(v):.4f}" for k, v in val_losses.items()])
        print(f"Epoch {epoch+1} Val - Loss: {avg_val_loss:.4f} - {val_loss_str}")
        
        # 早停检测
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, "best_aligner_model_checkpoint.pt")
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # 加载最佳模型
    checkpoint = torch.load("best_aligner_model_checkpoint.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    return model