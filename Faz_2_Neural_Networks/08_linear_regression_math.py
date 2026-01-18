"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DERS 08: LINEAR REGRESSION - SAF MATEMATİK İLE (nn.Module YASAK!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Amaç: Linear regression'ı sıfırdan, sadece tensor işlemleri ile yazmak.
Gradient descent'i manuel olarak implement etmek.

nn.Module, nn.Linear, optim.SGD YASAK! Sadece torch.tensor ve matematik.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np


def generate_data(n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sentetik linear data oluşturur: y = 3x + 2 + noise
    
    Args:
        n_samples: Örnek sayısı
        
    Returns:
        X: (n_samples, 1) - input
        y: (n_samples, 1) - target
    """
    torch.manual_seed(42)
    
    # X: 0 ile 10 arasında rastgele sayılar
    X = torch.rand(n_samples, 1) * 10
    
    # y = 3x + 2 + noise
    true_w = 3.0
    true_b = 2.0
    noise = torch.randn(n_samples, 1) * 0.5
    
    y = true_w * X + true_b + noise
    
    print(f"📊 Veri oluşturuldu:")
    print(f"   Gerçek w: {true_w}")
    print(f"   Gerçek b: {true_b}")
    print(f"   Örnekler: {n_samples}")
    
    return X, y


def initialize_parameters() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Model parametrelerini rastgele başlatır.
    
    Returns:
        w: (1, 1) - weight
        b: (1, 1) - bias
    """
    # Küçük rastgele değerlerle başlat
    w = torch.randn(1, 1, requires_grad=True)
    b = torch.zeros(1, 1, requires_grad=True)
    
    print(f"\n🎲 Parametreler başlatıldı:")
    print(f"   w: {w.item():.4f}")
    print(f"   b: {b.item():.4f}")
    
    return w, b


def forward(X: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Forward pass: y_pred = X @ w + b
    
    Args:
        X: (n, 1) - input
        w: (1, 1) - weight
        b: (1, 1) - bias
        
    Returns:
        y_pred: (n, 1) - predictions
    """
    return X @ w + b


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error loss.
    
    L = (1/n) * Σ(y_pred - y_true)²
    
    Args:
        y_pred: (n, 1) - predictions
        y_true: (n, 1) - targets
        
    Returns:
        loss: scalar
    """
    n = y_pred.shape[0]
    loss = ((y_pred - y_true) ** 2).sum() / n
    return loss


def train_manual_gradient(X: torch.Tensor, y: torch.Tensor, 
                          epochs: int = 100, lr: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Manuel gradient descent ile training (autograd YASAK!).
    
    Gradient formülleri:
    dL/dw = (2/n) * X^T @ (y_pred - y_true)
    dL/db = (2/n) * Σ(y_pred - y_true)
    
    Args:
        X: (n, 1) - input
        y: (n, 1) - target
        epochs: Epoch sayısı
        lr: Learning rate
        
    Returns:
        w, b, loss_history
    """
    print(f"\n🔧 MANUEL GRADIENT DESCENT")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {lr}")
    
    # Parametreleri başlat (requires_grad=False, manuel hesaplayacağız)
    w = torch.randn(1, 1)
    b = torch.zeros(1, 1)
    
    n = X.shape[0]
    loss_history = []
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = X @ w + b
        
        # Loss hesapla
        loss = ((y_pred - y) ** 2).sum() / n
        loss_history.append(loss.item())
        
        # Manuel gradient hesapla
        # dL/dw = (2/n) * X^T @ (y_pred - y)
        grad_w = (2.0 / n) * X.t() @ (y_pred - y)
        
        # dL/db = (2/n) * Σ(y_pred - y)
        grad_b = (2.0 / n) * (y_pred - y).sum()
        
        # Gradient descent update
        w = w - lr * grad_w
        b = b - lr * grad_b
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")
    
    return w, b, loss_history


def train_autograd(X: torch.Tensor, y: torch.Tensor, 
                   epochs: int = 100, lr: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    PyTorch autograd ile training.
    
    Args:
        X: (n, 1) - input
        y: (n, 1) - target
        epochs: Epoch sayısı
        lr: Learning rate
        
    Returns:
        w, b, loss_history
    """
    print(f"\n🤖 AUTOGRAD İLE TRAINING")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {lr}")
    
    # Parametreleri başlat (requires_grad=True)
    w = torch.randn(1, 1, requires_grad=True)
    b = torch.zeros(1, 1, requires_grad=True)
    
    loss_history = []
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = forward(X, w, b)
        
        # Loss hesapla
        loss = mse_loss(y_pred, y)
        loss_history.append(loss.item())
        
        # Backward pass (autograd!)
        loss.backward()
        
        # Gradient descent update (no_grad context'te)
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
            
            # Gradientleri sıfırla
            w.grad.zero_()
            b.grad.zero_()
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1:3d}: Loss = {loss.item():.4f}, w = {w.item():.4f}, b = {b.item():.4f}")
    
    return w, b, loss_history


def train_batch_gradient_descent(X: torch.Tensor, y: torch.Tensor,
                                  batch_size: int = 10, epochs: int = 100, 
                                  lr: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """
    Mini-batch gradient descent.
    
    Args:
        X: (n, 1) - input
        y: (n, 1) - target
        batch_size: Batch boyutu
        epochs: Epoch sayısı
        lr: Learning rate
        
    Returns:
        w, b, loss_history
    """
    print(f"\n📦 MINI-BATCH GRADIENT DESCENT")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {lr}")
    
    w = torch.randn(1, 1, requires_grad=True)
    b = torch.zeros(1, 1, requires_grad=True)
    
    n = X.shape[0]
    loss_history = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = torch.randperm(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0.0
        
        # Mini-batch training
        for i in range(0, n, batch_size):
            # Batch al
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward
            y_pred = forward(X_batch, w, b)
            loss = mse_loss(y_pred, y_batch)
            
            # Backward
            loss.backward()
            
            # Update
            with torch.no_grad():
                w -= lr * w.grad
                b -= lr * b.grad
                w.grad.zero_()
                b.grad.zero_()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (n // batch_size)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1:3d}: Loss = {avg_loss:.4f}, w = {w.item():.4f}, b = {b.item():.4f}")
    
    return w, b, loss_history


def visualize_results(X: torch.Tensor, y: torch.Tensor, 
                     w: torch.Tensor, b: torch.Tensor,
                     loss_history: list, title: str = "Linear Regression") -> None:
    """
    Sonuçları görselleştirir.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Sol: Data ve fitted line
    ax1.scatter(X.numpy(), y.numpy(), alpha=0.5, label='Data')
    
    X_line = torch.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = forward(X_line, w, b)
    
    ax1.plot(X_line.numpy(), y_line.detach().numpy(), 'r-', linewidth=2, label='Fitted line')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_title(f'{title}\nw={w.item():.2f}, b={b.item():.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sağ: Loss curve
    ax2.plot(loss_history)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'/tmp/{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Grafik kaydedildi: /tmp/{title.replace(' ', '_')}.png")


def demonstrate_gradient_comparison() -> None:
    """
    Manuel gradient vs autograd karşılaştırması.
    """
    print("\n" + "="*70)
    print("🔬 MANUEL GRADIENT VS AUTOGRAD KARŞILAŞTIRMASI")
    print("="*70)
    
    # Aynı veriyi kullan
    X, y = generate_data(100)
    
    # Manuel gradient
    w_manual, b_manual, loss_manual = train_manual_gradient(X, y, epochs=100, lr=0.01)
    
    # Autograd
    w_auto, b_auto, loss_auto = train_autograd(X, y, epochs=100, lr=0.01)
    
    # Karşılaştır
    print(f"\n📊 SONUÇLAR:")
    print(f"   Manuel:   w={w_manual.item():.4f}, b={b_manual.item():.4f}")
    print(f"   Autograd: w={w_auto.item():.4f}, b={b_auto.item():.4f}")
    print(f"   Fark:     w={abs(w_manual.item() - w_auto.item()):.6f}, b={abs(b_manual.item() - b_auto.item()):.6f}")
    
    visualize_results(X, y, w_auto, b_auto, loss_auto, "Autograd Training")


def demonstrate_batch_sizes() -> None:
    """
    Farklı batch size'ların etkisini gösterir.
    """
    print("\n" + "="*70)
    print("📦 BATCH SIZE ETKİSİ")
    print("="*70)
    
    X, y = generate_data(100)
    
    batch_sizes = [1, 10, 50, 100]
    
    for bs in batch_sizes:
        print(f"\n{'─'*70}")
        w, b, loss_hist = train_batch_gradient_descent(X, y, batch_size=bs, epochs=50, lr=0.01)
        print(f"   Final: w={w.item():.4f}, b={b.item():.4f}, loss={loss_hist[-1]:.4f}")


def main() -> None:
    """
    Ana çalıştırma fonksiyonu.
    """
    print("\n" + "="*70)
    print("🚀 LINEAR REGRESSION - SAF MATEMATİK İLE")
    print("="*70)
    
    demonstrate_gradient_comparison()
    demonstrate_batch_sizes()
    
    print("\n" + "="*70)
    print("✅ DERS 08 TAMAMLANDI!")
    print("="*70 + "\n")
    
    print("💡 Öğrendikleriniz:")
    print("   - Manuel gradient hesaplama")
    print("   - Autograd ile karşılaştırma")
    print("   - Batch vs mini-batch vs SGD")
    print("   - Loss curve analizi")


if __name__ == "__main__":
    main()
