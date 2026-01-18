"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DERS 10: ACTIVATION FUNCTIONS - ReLU, GELU, Swish VE VANISHING GRADIENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Amaç: Activation function'ların matematiksel özelliklerini anlamak.
Vanishing/exploding gradient problemini çözmek.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable


def plot_activation(func: Callable, name: str, x_range: tuple = (-5, 5)) -> None:
    """Activation function ve türevini çizer."""
    x = torch.linspace(x_range[0], x_range[1], 1000, requires_grad=True)
    y = func(x)
    
    # Türev hesapla
    y.sum().backward()
    grad = x.grad
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Fonksiyon
    ax1.plot(x.detach().numpy(), y.detach().numpy(), linewidth=2)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax1.set_title(f'{name}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.grid(True, alpha=0.3)
    
    # Türev
    ax2.plot(x.detach().numpy(), grad.numpy(), linewidth=2, color='red')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax2.set_title(f"{name} - Türev")
    ax2.set_xlabel('x')
    ax2.set_ylabel("f'(x)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'/tmp/{name.lower().replace(" ", "_")}.png', dpi=150)
    print(f"📊 {name} grafiği kaydedildi")


def demonstrate_sigmoid() -> None:
    """Sigmoid: σ(x) = 1 / (1 + e^(-x))"""
    print("\n" + "🎯 SIGMOID".center(70, "━"))
    
    x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
    y = torch.sigmoid(x)
    
    print(f"Input:  {x.tolist()}")
    print(f"Output: {[f'{v:.4f}' for v in y.tolist()]}")
    print(f"\n✅ Avantajlar: Smooth, (0,1) aralığı")
    print(f"❌ Dezavantajlar: Vanishing gradient (x→±∞ için türev→0)")
    
    plot_activation(torch.sigmoid, "Sigmoid")


def demonstrate_tanh() -> None:
    """Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))"""
    print("\n" + "🎯 TANH".center(70, "━"))
    
    x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
    y = torch.tanh(x)
    
    print(f"Input:  {x.tolist()}")
    print(f"Output: {[f'{v:.4f}' for v in y.tolist()]}")
    print(f"\n✅ Avantajlar: Zero-centered, sigmoid'den iyi")
    print(f"❌ Dezavantajlar: Hala vanishing gradient var")
    
    plot_activation(torch.tanh, "Tanh")


def demonstrate_relu() -> None:
    """ReLU: f(x) = max(0, x)"""
    print("\n" + "🎯 RELU".center(70, "━"))
    
    x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
    y = F.relu(x)
    
    print(f"Input:  {x.tolist()}")
    print(f"Output: {y.tolist()}")
    print(f"\n✅ Avantajlar: Hızlı, vanishing gradient yok (x>0)")
    print(f"❌ Dezavantajlar: Dying ReLU (x<0 için gradient=0)")
    
    plot_activation(F.relu, "ReLU")


def demonstrate_leaky_relu() -> None:
    """Leaky ReLU: f(x) = max(0.01x, x)"""
    print("\n" + "🎯 LEAKY RELU".center(70, "━"))
    
    x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
    y = F.leaky_relu(x, negative_slope=0.01)
    
    print(f"Input:  {x.tolist()}")
    print(f"Output: {[f'{v:.4f}' for v in y.tolist()]}")
    print(f"\n✅ Avantajlar: Dying ReLU problemi yok")
    print(f"❌ Dezavantajlar: Negatif slope hyperparameter")


def demonstrate_gelu() -> None:
    """GELU: Gaussian Error Linear Unit"""
    print("\n" + "🎯 GELU (Transformer'larda kullanılır)".center(70, "━"))
    
    x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
    y = F.gelu(x)
    
    print(f"Input:  {x.tolist()}")
    print(f"Output: {[f'{v:.4f}' for v in y.tolist()]}")
    print(f"\n✅ Avantajlar: Smooth, probabilistic, BERT/GPT'de kullanılır")
    print(f"💡 Formül: x * Φ(x) (Φ: Gaussian CDF)")
    
    plot_activation(F.gelu, "GELU")


def demonstrate_swish() -> None:
    """Swish (SiLU): f(x) = x * sigmoid(x)"""
    print("\n" + "🎯 SWISH / SiLU".center(70, "━"))
    
    x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
    y = F.silu(x)  # Swish = SiLU
    
    print(f"Input:  {x.tolist()}")
    print(f"Output: {[f'{v:.4f}' for v in y.tolist()]}")
    print(f"\n✅ Avantajlar: Self-gated, smooth, EfficientNet'te kullanılır")
    print(f"💡 Formül: x * σ(x)")
    
    plot_activation(F.silu, "Swish (SiLU)")


def demonstrate_vanishing_gradient() -> None:
    """Vanishing gradient problemini gösterir."""
    print("\n" + "🎯 VANISHING GRADIENT PROBLEMİ".center(70, "━"))
    
    # Derin sigmoid network
    class DeepSigmoid(nn.Module):
        def __init__(self, depth: int = 10):
            super().__init__()
            layers = []
            for _ in range(depth):
                layers.append(nn.Linear(10, 10))
                layers.append(nn.Sigmoid())
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    # Test
    model = DeepSigmoid(depth=10)
    x = torch.randn(1, 10, requires_grad=True)
    y = model(x)
    y.sum().backward()
    
    print(f"\n📊 10 Katmanlı Sigmoid Network:")
    print(f"   Input gradient norm: {x.grad.norm().item():.2e}")
    
    # Katman gradientlerini incele
    print(f"\n   Katman Gradientleri:")
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.grad is not None and 'weight' in name:
            print(f"   Layer {i//2}: {param.grad.norm().item():.2e}")
    
    print(f"\n❌ İlk katmanların gradienti çok küçük (vanishing)!")


def demonstrate_activation_comparison() -> None:
    """Farklı activation'ları karşılaştırır."""
    print("\n" + "🎯 ACTIVATION KARŞILAŞTIRMASI".center(70, "━"))
    
    x = torch.linspace(-5, 5, 100)
    
    activations = {
        'Sigmoid': torch.sigmoid(x),
        'Tanh': torch.tanh(x),
        'ReLU': F.relu(x),
        'GELU': F.gelu(x),
        'Swish': F.silu(x)
    }
    
    plt.figure(figsize=(10, 6))
    for name, y in activations.items():
        plt.plot(x.numpy(), y.numpy(), label=name, linewidth=2)
    
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Activation Functions Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/tmp/activation_comparison.png', dpi=150)
    print(f"\n📊 Karşılaştırma grafiği kaydedildi")


def main() -> None:
    print("\n" + "="*70)
    print("🚀 ACTIVATION FUNCTIONS")
    print("="*70)
    
    demonstrate_sigmoid()
    demonstrate_tanh()
    demonstrate_relu()
    demonstrate_leaky_relu()
    demonstrate_gelu()
    demonstrate_swish()
    demonstrate_vanishing_gradient()
    demonstrate_activation_comparison()
    
    print("\n" + "="*70)
    print("✅ DERS 10 TAMAMLANDI!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
