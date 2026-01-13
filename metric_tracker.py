"""
Metrics tracking and visualization during training
保存训练和验证指标，每3个epoch更新一次可视化图表
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


class MetricsTracker:
    """Track and visualize training metrics"""
    
    def __init__(self, save_dir, plot_interval=3):
        """
        save_dir: 保存目录
        plot_interval: 每N个epoch绘制一次图表
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plot_interval = plot_interval
        
        # 存储指标
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.epochs = []
        
        # 定义指标列表
        self.train_keys = ['total', 'coarse', 'refine', 'edge', 'corner']
        self.val_keys = ['psnr_mask', 'ssim_mask', 'psnr_boundary']
    
    def log_train_metrics(self, epoch, metrics_dict):
        """
        记录训练指标
        metrics_dict: {'total': val, 'coarse': val, 'refine': val, 'edge': val, 'corner': val}
        """
        if epoch not in self.epochs:
            self.epochs.append(epoch)
        
        for key in self.train_keys:
            value = metrics_dict.get(key, 0.0)
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.train_metrics[key].append(float(value))
    
    def log_val_metrics(self, epoch, metrics_dict):
        """
        记录验证指标
        metrics_dict: {'psnr_mask': val, 'ssim_mask': val, 'psnr_boundary': val, ...}
        """
        for key in self.val_keys:
            value = metrics_dict.get(key, 0.0)
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.val_metrics[key].append(float(value))
    
    def plot_metrics(self, epoch):
        """
        每plot_interval个epoch绘制一次图表
        epoch: 当前epoch
        """
        if len(self.epochs) == 0 or (epoch + 1) % self.plot_interval != 0:
            return
        
        # 创建训练指标图 (5个损失)
        self._plot_train_metrics(epoch)
        
        # 创建验证指标图 (3个指标)
        self._plot_val_metrics(epoch)
        
        # 保存数据
        self._save_csv(epoch)
    
    def _plot_train_metrics(self, epoch):
        """绘制5个训练损失"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training Metrics - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        # Flatten axes
        axes = axes.flatten()
        
        # 绘制每个指标
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, key in enumerate(self.train_keys):
            if idx < len(axes):
                ax = axes[idx]
                if len(self.train_metrics[key]) > 0:
                    ax.plot(self.train_metrics[key], color=colors[idx], linewidth=2, marker='o', markersize=4)
                    ax.set_title(f'{key.upper()} Loss', fontweight='bold')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.grid(True, alpha=0.3)
                    
                    # 显示最后的值
                    last_val = self.train_metrics[key][-1]
                    ax.text(0.98, 0.97, f'Last: {last_val:.4f}', 
                           transform=ax.transAxes, ha='right', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 隐藏多余的subplot
        for idx in range(len(self.train_keys), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'train_metrics.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Train metrics plot saved at epoch {epoch}")
    
    def _plot_val_metrics(self, epoch):
        """绘制3个验证指标"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'Validation Metrics - Epoch {epoch}', fontsize=16, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for idx, key in enumerate(self.val_keys):
            ax = axes[idx]
            if len(self.val_metrics[key]) > 0:
                ax.plot(self.val_metrics[key], color=colors[idx], linewidth=2, marker='s', markersize=5)
                ax.set_title(key.replace('_', ' ').upper(), fontweight='bold')
                ax.set_xlabel('Validation Step')
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)
                
                # 显示最大值和当前值
                max_val = max(self.val_metrics[key])
                last_val = self.val_metrics[key][-1]
                
                info_text = f'Max: {max_val:.4f}\nLast: {last_val:.4f}'
                ax.text(0.98, 0.97, info_text, 
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'val_metrics.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Val metrics plot saved at epoch {epoch}")
    
    def _save_csv(self, epoch):
        """保存指标到CSV文件便于后续分析"""
        import csv
        
        csv_path = self.save_dir / 'metrics.csv'
        
        # 写入CSV
        with open(csv_path, 'w', newline='') as f:
            # 确定总行数（最长的列表）
            max_len = max(
                len(self.train_metrics.get('total', [])),
                len(self.val_metrics.get('psnr_mask', []))
            )
            
            # 写入表头
            headers = ['Epoch'] + [f'train_{k}' for k in self.train_keys] + [f'val_{k}' for k in self.val_keys]
            writer = csv.writer(f)
            writer.writerow(headers)
            
            # 写入数据
            for i in range(max_len):
                row = [i]
                
                # 训练指标
                for key in self.train_keys:
                    val = self.train_metrics[key][i] if i < len(self.train_metrics[key]) else ''
                    row.append(f'{val:.6f}' if isinstance(val, float) else val)
                
                # 验证指标
                for key in self.val_keys:
                    val = self.val_metrics[key][i] if i < len(self.val_metrics[key]) else ''
                    row.append(f'{val:.6f}' if isinstance(val, float) else val)
                
                writer.writerow(row)
        
        print(f"✓ Metrics saved to CSV at epoch {epoch}")
    
    def get_summary(self):
        """获取指标摘要"""
        summary = {}
        
        for key in self.train_keys:
            if len(self.train_metrics[key]) > 0:
                summary[f'train_{key}'] = {
                    'current': self.train_metrics[key][-1],
                    'min': min(self.train_metrics[key]),
                    'max': max(self.train_metrics[key])
                }
        
        for key in self.val_keys:
            if len(self.val_metrics[key]) > 0:
                summary[f'val_{key}'] = {
                    'current': self.val_metrics[key][-1],
                    'min': min(self.val_metrics[key]),
                    'max': max(self.val_metrics[key])
                }
        
        return summary
    
    def print_summary(self):
        """打印指标摘要"""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("METRICS SUMMARY")
        print("="*70)
        
        print("\nTRAINING METRICS:")
        for key in self.train_keys:
            key_full = f'train_{key}'
            if key_full in summary:
                s = summary[key_full]
                print(f"  {key:10s}: curr={s['current']:8.4f}, min={s['min']:8.4f}, max={s['max']:8.4f}")
        
        print("\nVALIDATION METRICS:")
        for key in self.val_keys:
            key_full = f'val_{key}'
            if key_full in summary:
                s = summary[key_full]
                print(f"  {key:15s}: curr={s['current']:8.4f}, min={s['min']:8.4f}, max={s['max']:8.4f}")
        
        print("="*70)


if __name__ == "__main__":
    # 测试 MetricsTracker
    tracker = MetricsTracker('./test_tracking')
    
    # 模拟训练数据
    for epoch in range(20):
        train_metrics = {
            'total': 0.05 - 0.001*epoch + np.random.randn()*0.002,
            'coarse': 0.02 - 0.0005*epoch + np.random.randn()*0.001,
            'refine': 0.015 - 0.0003*epoch + np.random.randn()*0.0008,
            'edge': 0.008 + np.random.randn()*0.0005,
            'corner': 0.005 + np.random.randn()*0.0003
        }
        tracker.log_train_metrics(epoch, train_metrics)
        
        if (epoch + 1) % 3 == 0:
            val_metrics = {
                'psnr_mask': 25 + 0.5*epoch + np.random.randn()*0.5,
                'ssim_mask': 0.8 + 0.003*epoch + np.random.randn()*0.01,
                'psnr_boundary': 24 + 0.4*epoch + np.random.randn()*0.4
            }
            tracker.log_val_metrics(epoch, val_metrics)
            tracker.plot_metrics(epoch)
    
    tracker.print_summary()
    print(f"\n✓ Test plots saved to ./test_tracking/")