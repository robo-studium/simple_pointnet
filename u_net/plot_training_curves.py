import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(csv_path, output_dir=None, dpi=300):
    """
    CSVファイルから学習曲線を描画して保存
    
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
    plt.rcParams['font.size'] = 10
    
    # メインの学習曲線プロット (2x3)
    # fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # epochs = df['epoch']
    
    # # 1. Loss
    # axes[0, 0].plot(epochs, df['train_loss'], 'b-', label='Train', linewidth=2, marker='o', markersize=3)
    # axes[0, 0].plot(epochs, df['val_loss'], 'r-', label='Val', linewidth=2, marker='s', markersize=3)
    # axes[0, 0].set_xlabel('Epoch', fontsize=11)
    # axes[0, 0].set_ylabel('Loss', fontsize=11)
    # axes[0, 0].set_title('Loss', fontsize=12, fontweight='bold')
    # axes[0, 0].legend(fontsize=10)
    # axes[0, 0].grid(True, alpha=0.3)


    # 1つのグラフだけ描画する場合（Lossのみの例）
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))  # サイズは好みで調整（例: 幅8、高さ6）

    fig.suptitle('Training History', fontsize=16, fontweight='bold')

    epochs = df['epoch']

    # Lossのプロット
    ax.plot(epochs, df['train_f1'], 'b-', label='Train', linewidth=2, marker='o', markersize=3)
    ax.plot(epochs, df['val_f1'], 'r-', label='Val', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('F1 Score', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()  # タイトルとサブプロットの重なりを防ぐ
    plt.show()
    
    # 2. Accuracy
    # axes[0, 1].plot(epochs, df['train_accuracy'], 'b-', label='Train', linewidth=2, marker='o', markersize=3)
    # axes[0, 1].plot(epochs, df['val_accuracy'], 'r-', label='Val', linewidth=2, marker='s', markersize=3)
    # axes[0, 1].set_xlabel('Epoch', fontsize=11)
    # axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    # axes[0, 1].set_title('Accuracy', fontsize=12, fontweight='bold')
    # axes[0, 1].legend(fontsize=10)
    # axes[0, 1].grid(True, alpha=0.3)
    # axes[0, 1].set_ylim([0, 1])
    
    # 3. F1 Score
    # axes[0, 2].plot(epochs, df['train_f1'], 'b-', label='Train', linewidth=2, marker='o', markersize=3)
    # axes[0, 2].plot(epochs, df['val_f1'], 'r-', label='Val', linewidth=2, marker='s', markersize=3)
    # axes[0, 2].set_xlabel('Epoch', fontsize=11)
    # axes[0, 2].set_ylabel('F1 Score', fontsize=11)
    # axes[0, 2].set_title('F1 Score', fontsize=12, fontweight='bold')
    # axes[0, 2].legend(fontsize=10)
    # axes[0, 2].grid(True, alpha=0.3)
    # axes[0, 2].set_ylim([0, 1])
    
    # 4. Precision
    # axes[1, 0].plot(epochs, df['train_precision'], 'b-', label='Train', linewidth=2, marker='o', markersize=3)
    # axes[1, 0].plot(epochs, df['val_precision'], 'r-', label='Val', linewidth=2, marker='s', markersize=3)
    # axes[1, 0].set_xlabel('Epoch', fontsize=11)
    # axes[1, 0].set_ylabel('Precision', fontsize=11)
    # axes[1, 0].set_title('Precision', fontsize=12, fontweight='bold')
    # axes[1, 0].legend(fontsize=10)
    # axes[1, 0].grid(True, alpha=0.3)
    # axes[1, 0].set_ylim([0, 1])
    
    # 5. Recall
    # axes[1, 1].plot(epochs, df['train_recall'], 'b-', label='Train', linewidth=2, marker='o', markersize=3)
    # axes[1, 1].plot(epochs, df['val_recall'], 'r-', label='Val', linewidth=2, marker='s', markersize=3)
    # axes[1, 1].set_xlabel('Epoch', fontsize=11)
    # axes[1, 1].set_ylabel('Recall', fontsize=11)
    # axes[1, 1].set_title('Recall', fontsize=12, fontweight='bold')
    # axes[1, 1].legend(fontsize=10)
    # axes[1, 1].grid(True, alpha=0.3)
    # axes[1, 1].set_ylim([0, 1])
    
    # 6. IoU (Validation only)
    # axes[1, 2].plot(epochs, df['val_iou_noise'], 'g-', label='IoU Noise', linewidth=2, marker='o', markersize=3)
    # axes[1, 2].plot(epochs, df['val_iou_clean'], 'orange', label='IoU Clean', linewidth=2, marker='s', markersize=3)
    # axes[1, 2].set_xlabel('Epoch', fontsize=11)
    # axes[1, 2].set_ylabel('IoU', fontsize=11)
    # axes[1, 2].set_title('Validation IoU', fontsize=12, fontweight='bold')
    # axes[1, 2].legend(fontsize=10)
    # axes[1, 2].grid(True, alpha=0.3)
    # axes[1, 2].set_ylim([0, 1])
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()
    
    # 追加: Learning Rate のプロット
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
    
    # 追加: Loss の拡大図 (最初の数エポックを除外)
    if len(df) > 10:
        fig, ax = plt.subplots(figsize=(10, 6))
        skip_epochs = 5  # 最初の5エポックをスキップ
        ax.plot(epochs[skip_epochs:], df['train_loss'][skip_epochs:], 'b-', 
                label='Train', linewidth=2, marker='o', markersize=3)
        ax.plot(epochs[skip_epochs:], df['val_loss'][skip_epochs:], 'r-', 
                label='Val', linewidth=2, marker='s', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Loss (from Epoch {skip_epochs+1})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        loss_zoom_path = os.path.join(output_dir, 'loss_zoom.png')
        plt.savefig(loss_zoom_path, dpi=dpi, bbox_inches='tight')
        print(f"Zoomed loss plot saved to: {loss_zoom_path}")
        plt.close()
    
    # 統計情報の出力
    print("\n=== Training Statistics ===")
    print(f"Best Validation F1: {df['val_f1'].max():.4f} at epoch {df.loc[df['val_f1'].idxmax(), 'epoch']:.0f}")
    print(f"Best Validation Accuracy: {df['val_accuracy'].max():.4f} at epoch {df.loc[df['val_accuracy'].idxmax(), 'epoch']:.0f}")
    print(f"Lowest Validation Loss: {df['val_loss'].min():.4f} at epoch {df.loc[df['val_loss'].idxmin(), 'epoch']:.0f}")
    print(f"Final Validation F1: {df['val_f1'].iloc[-1]:.4f}")
    print(f"Final Validation Accuracy: {df['val_accuracy'].iloc[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Plot training curves from CSV log')
    parser.add_argument(
        '--csv_path',
        type=str,
        default='./outputs/training_log.csv',
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