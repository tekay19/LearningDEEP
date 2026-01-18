"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DERS 11: LOSS FUNCTIONS - MSE, CrossEntropy, Focal Loss
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Amaç: Loss function'ların matematiksel özelliklerini anlamak.
Classification vs regression loss'ları karşılaştırmak.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def demonstrate_mse_loss() -> None:
    """Mean Squared Error - Regression için"""
    print("\n" + "🎯 MSE LOSS (Regression)".center(70, "━"))
    
    pred = torch.tensor([2.5, 3.0, 4.2])
    target = torch.tensor([2.0, 3.5, 4.0])
    
    # Manuel hesaplama
    mse_manual = ((pred - target) ** 2).mean()
    
    # PyTorch
    mse_torch = F.mse_loss(pred, target)
    
    print(f"Predictions: {pred.tolist()}")
    print(f"Targets:     {target.tolist()}")
    print(f"MSE (manuel): {mse_manual.item():.4f}")
    print(f"MSE (torch):  {mse_torch.item():.4f}")
    print(f"\n💡 Formül: (1/n) Σ(y_pred - y_true)²")


def demonstrate_mae_loss() -> None:
    """Mean Absolute Error - Outlier'lara robust"""
    print("\n" + "🎯 MAE LOSS (L1 Loss)".center(70, "━"))
    
    pred = torch.tensor([2.5, 3.0, 10.0])  # 10.0 outlier
    target = torch.tensor([2.0, 3.5, 4.0])
    
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    
    print(f"Predictions: {pred.tolist()}")
    print(f"Targets:     {target.tolist()}")
    print(f"MSE: {mse.item():.4f}")
    print(f"MAE: {mae.item():.4f}")
    print(f"\n💡 MAE outlier'lara daha az duyarlı")


def demonstrate_bce_loss() -> None:
    """Binary Cross Entropy - Binary classification"""
    print("\n" + "🎯 BCE LOSS (Binary Classification)".center(70, "━"))
    
    # Sigmoid çıktısı (0-1 arası)
    pred = torch.tensor([0.9, 0.3, 0.8])
    target = torch.tensor([1.0, 0.0, 1.0])
    
    bce = F.binary_cross_entropy(pred, target)
    
    print(f"Predictions: {pred.tolist()}")
    print(f"Targets:     {target.tolist()}")
    print(f"BCE: {bce.item():.4f}")
    print(f"\n💡 Formül: -[y*log(p) + (1-y)*log(1-p)]")


def demonstrate_crossentropy_loss() -> None:
    """Cross Entropy - Multi-class classification"""
    print("\n" + "🎯 CROSS ENTROPY LOSS (Multi-class)".center(70, "━"))
    
    # Logits (softmax öncesi)
    logits = torch.tensor([[2.0, 1.0, 0.1],
                           [0.5, 2.5, 0.3]])
    target = torch.tensor([0, 1])  # Class indices
    
    ce = F.cross_entropy(logits, target)
    
    print(f"Logits:\n{logits}")
    print(f"Targets: {target.tolist()}")
    print(f"Cross Entropy: {ce.item():.4f}")
    
    # Manuel hesaplama
    print(f"\n🔍 Manuel Hesaplama:")
    probs = F.softmax(logits, dim=1)
    print(f"Softmax probabilities:\n{probs}")
    
    # İlk örnek için loss
    loss_0 = -torch.log(probs[0, target[0]])
    print(f"Sample 0 loss: -log({probs[0, target[0]].item():.4f}) = {loss_0.item():.4f}")


def demonstrate_nll_loss() -> None:
    """Negative Log Likelihood - CrossEntropy'nin parçası"""
    print("\n" + "🎯 NLL LOSS".center(70, "━"))
    
    # Log probabilities
    log_probs = torch.tensor([[-0.5, -1.2, -2.3],
                              [-1.5, -0.3, -2.0]])
    target = torch.tensor([0, 1])
    
    nll = F.nll_loss(log_probs, target)
    
    print(f"Log Probs:\n{log_probs}")
    print(f"Targets: {target.tolist()}")
    print(f"NLL: {nll.item():.4f}")
    print(f"\n💡 CrossEntropy = LogSoftmax + NLL")


def demonstrate_focal_loss() -> None:
    """Focal Loss - Imbalanced classification için"""
    print("\n" + "🎯 FOCAL LOSS (Imbalanced Data)".center(70, "━"))
    
    def focal_loss(pred, target, alpha=0.25, gamma=2.0):
        """
        Focal Loss = -α(1-p)^γ log(p)
        
        Args:
            pred: Predictions (after sigmoid)
            target: Ground truth (0 or 1)
            alpha: Balancing factor
            gamma: Focusing parameter
        """
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)  # p_t
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()
    
    # Kolay örnek (high confidence)
    easy_pred = torch.tensor([0.95])
    easy_target = torch.tensor([1.0])
    
    # Zor örnek (low confidence)
    hard_pred = torch.tensor([0.55])
    hard_target = torch.tensor([1.0])
    
    bce_easy = F.binary_cross_entropy(easy_pred, easy_target)
    bce_hard = F.binary_cross_entropy(hard_pred, hard_target)
    
    focal_easy = focal_loss(easy_pred, easy_target)
    focal_hard = focal_loss(hard_pred, hard_target)
    
    print(f"Kolay örnek (p=0.95):")
    print(f"  BCE: {bce_easy.item():.4f}, Focal: {focal_easy.item():.4f}")
    
    print(f"\nZor örnek (p=0.55):")
    print(f"  BCE: {bce_hard.item():.4f}, Focal: {focal_hard.item():.4f}")
    
    print(f"\n💡 Focal loss zor örneklere daha fazla odaklanır")


def demonstrate_huber_loss() -> None:
    """Huber Loss - MSE ve MAE arası"""
    print("\n" + "🎯 HUBER LOSS (Robust Regression)".center(70, "━"))
    
    pred = torch.tensor([1.0, 2.0, 10.0])
    target = torch.tensor([1.5, 2.5, 3.0])
    
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    huber = F.smooth_l1_loss(pred, target)
    
    print(f"Predictions: {pred.tolist()}")
    print(f"Targets:     {target.tolist()}")
    print(f"MSE:   {mse.item():.4f}")
    print(f"MAE:   {mae.item():.4f}")
    print(f"Huber: {huber.item():.4f}")
    print(f"\n💡 Küçük hatalarda MSE, büyük hatalarda MAE gibi davranır")


def main() -> None:
    print("\n" + "="*70)
    print("🚀 LOSS FUNCTIONS")
    print("="*70)
    
    demonstrate_mse_loss()
    demonstrate_mae_loss()
    demonstrate_bce_loss()
    demonstrate_crossentropy_loss()
    demonstrate_nll_loss()
    demonstrate_focal_loss()
    demonstrate_huber_loss()
    
    print("\n" + "="*70)
    print("✅ DERS 11 TAMAMLANDI!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
