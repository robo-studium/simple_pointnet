import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


def plot_training_curves(csv_path, output_dir=None, dpi=300):
    """
    CSVファイルから学習曲線を描画して保存（各グラフを個別ファイルとして）
    
    Args:
        csv_path: training_log.csvへのパス
        output_dir: 保存先ディレクトリ (Noneの場合はCSVと同じディレクトリ)
        dpi: 画像の解像度
    """
    # CSVファイルの読み込み
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} epochs from {csv_path}")
    
    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # スタイル設定
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11
    
    epochs = df['epoch']
    
    # 1. Total Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['train_total_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, df['val_total_loss'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Total Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'total_loss.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Total Loss plot saved to: {save_path}")
    plt.close()
    
    # 2. Task Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['train_task_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, df['val_task_loss'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Task Loss', fontsize=12)
    ax.set_title('Task Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'task_loss.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Task Loss plot saved to: {save_path}")
    plt.close()
    
    # 3. Distillation Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['train_distill_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, df['val_distill_loss'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Distillation Loss', fontsize=12)
    ax.set_title('Distillation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'distill_loss.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Distillation Loss plot saved to: {save_path}")
    plt.close()
    
    # 4. Feature Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['train_feature_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, df['val_feature_loss'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Feature Loss', fontsize=12)
    ax.set_title('Feature Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'feature_loss.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Feature Loss plot saved to: {save_path}")
    plt.close()
    
    # 5. Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['train_accuracy'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, df['val_accuracy'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'accuracy.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Accuracy plot saved to: {save_path}")
    plt.close()
    
    # 6. F1 Score
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['train_f1'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, df['val_f1'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'f1_score.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"F1 Score plot saved to: {save_path}")
    plt.close()
    
    # 7. Precision
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['train_precision'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, df['val_precision'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'precision.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Precision plot saved to: {save_path}")
    plt.close()
    
    # 8. Recall
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['train_recall'], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, df['val_recall'], 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Recall', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'recall.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Recall plot saved to: {save_path}")
    plt.close()
    
    # 9. IoU (Train & Val)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['train_iou_noise'], 'b-', label='Train IoU Noise', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, df['train_iou_clean'], 'c-', label='Train IoU Clean', linewidth=2, marker='^', markersize=4)
    ax.plot(epochs, df['val_iou_noise'], 'r-', label='Val IoU Noise', linewidth=2, marker='s', markersize=4)
    ax.plot(epochs, df['val_iou_clean'], 'orange', label='Val IoU Clean', linewidth=2, marker='D', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title('IoU (Noise & Clean)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'iou.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"IoU plot saved to: {save_path}")
    plt.close()
    
    # 10. Learning Rate
    if 'lr' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, df['lr'], 'purple', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        lr_save_path = os.path.join(output_dir, 'learning_rate.png')
        plt.savefig(lr_save_path, dpi=dpi, bbox_inches='tight')
        print(f"Learning rate plot saved to: {lr_save_path}")
        plt.close()
    
    # 11. Total Loss の拡大図 (最初の数エポックを除外)
    if len(df) > 10:
        fig, ax = plt.subplots(figsize=(10, 6))
        skip_epochs = 5  # 最初の5エポックをスキップ
        ax.plot(epochs[skip_epochs:], df['train_total_loss'][skip_epochs:], 'b-', 
                label='Train', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs[skip_epochs:], df['val_total_loss'][skip_epochs:], 'r-', 
                label='Val', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Total Loss', fontsize=12)
        ax.set_title(f'Total Loss (from Epoch {skip_epochs+1})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_zoom_path = os.path.join(output_dir, 'total_loss_zoom.png')
        plt.savefig(loss_zoom_path, dpi=dpi, bbox_inches='tight')
        print(f"Zoomed total loss plot saved to: {loss_zoom_path}")
        plt.close()
    
    # 統計情報の出力
    print("\n=== Training Statistics ===")
    print(f"Best Validation F1: {df['val_f1'].max():.4f} at epoch {df.loc[df['val_f1'].idxmax(), 'epoch']:.0f}")
    print(f"Best Validation Accuracy: {df['val_accuracy'].max():.4f} at epoch {df.loc[df['val_accuracy'].idxmax(), 'epoch']:.0f}")
    print(f"Lowest Validation Total Loss: {df['val_total_loss'].min():.4f} at epoch {df.loc[df['val_total_loss'].idxmin(), 'epoch']:.0f}")
    print(f"Best Validation IoU Noise: {df['val_iou_noise'].max():.4f} at epoch {df.loc[df['val_iou_noise'].idxmax(), 'epoch']:.0f}")
    print(f"Best Validation IoU Clean: {df['val_iou_clean'].max():.4f} at epoch {df.loc[df['val_iou_clean'].idxmax(), 'epoch']:.0f}")
    print(f"Final Validation F1: {df['val_f1'].iloc[-1]:.4f}")
    print(f"Final Validation Accuracy: {df['val_accuracy'].iloc[-1]:.4f}")
    print(f"Final Validation IoU Noise: {df['val_iou_noise'].iloc[-1]:.4f}")
    print(f"Final Validation IoU Clean: {df['val_iou_clean'].iloc[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Plot training curves from CSV log')
    parser.add_argument(
        '--csv_path',
        type=str,
        default='./outputs_student/distillation_log.csv',
        help='Path to training_log.csv'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for plots (default: same as CSV directory)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved images'
    )
    
    args = parser.parse_args()
    
    plot_training_curves(args.csv_path, args.output_dir, args.dpi)


if __name__ == '__main__':
    main()