"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DERS 12: OPTIMIZER ALGORITHMS - SGD, Momentum, Adam, AdamW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Amaç: Optimizer algoritmalarının matematiksel özelliklerini anlamak.
SGD'den AdamW'ye geçiş sürecini öğrenmek.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.optim as optim


def demonstrate_sgd() -> None:
    """Stochastic Gradient Descent"""
    print("\n" + "🎯 SGD (Stochastic Gradient Descent)".center(70, "━"))
    
    # Basit model
    model = nn.Linear(2, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    print(f"💡 Update rule: θ = θ - lr * ∇θ")
    print(f"   Learning rate: 0.01")
    
    # Dummy forward-backward
    x = torch.randn(10, 2)
    y = torch.randn(10, 1)
    
    pred = model(x)
    loss = F.mse_loss(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    print(f"\nÖnceki weight: {model.weight.data[0, 0].item():.4f}")
    optimizer.step()
    print(f"Sonraki weight: {model.weight.data[0, 0].item():.4f}")


def demonstrate_momentum() -> None:
    """SGD with Momentum"""
    print("\n" + "🎯 MOMENTUM".center(70, "━"))
    
    model = nn.Linear(2, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    print(f"💡 Update rule:")
    print(f"   v_t = β*v_(t-1) + ∇θ")
    print(f"   θ = θ - lr * v_t")
    print(f"\n   Momentum: 0.9")
    print(f"   ✅ Avantaj: Oscillation azalır, hızlanır")


def demonstrate_rmsprop() -> None:
    """RMSProp - Adaptive learning rate"""
    print("\n" + "🎯 RMSPROP".center(70, "━"))
    
    model = nn.Linear(2, 1)
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    
    print(f"💡 Update rule:")
    print(f"   E[g²]_t = β*E[g²]_(t-1) + (1-β)*g²")
    print(f"   θ = θ - lr * g / √(E[g²] + ε)")
    print(f"\n   ✅ Her parametreye farklı learning rate")


def demonstrate_adam() -> None:
    """Adam - Adaptive Moment Estimation"""
    print("\n" + "🎯 ADAM (En popüler!)".center(70, "━"))
    
    model = nn.Linear(2, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    
    print(f"💡 Update rule:")
    print(f"   m_t = β1*m_(t-1) + (1-β1)*g      (1st moment)")
    print(f"   v_t = β2*v_(t-1) + (1-β2)*g²     (2nd moment)")
    print(f"   m̂ = m_t / (1-β1^t)               (bias correction)")
    print(f"   v̂ = v_t / (1-β2^t)")
    print(f"   θ = θ - lr * m̂ / (√v̂ + ε)")
    print(f"\n   β1=0.9, β2=0.999")
    print(f"   ✅ Momentum + RMSProp birleşimi")


def demonstrate_adamw() -> None:
    """AdamW - Adam with decoupled weight decay"""
    print("\n" + "🎯 ADAMW (Transformer'larda standart)".center(70, "━"))
    
    model = nn.Linear(2, 1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    print(f"💡 Update rule:")
    print(f"   Adam update + weight decay:")
    print(f"   θ = θ - lr * m̂ / (√v̂ + ε) - lr * λ * θ")
    print(f"\n   Weight decay: 0.01")
    print(f"   ✅ L2 regularization'dan daha iyi")
    print(f"   ✅ BERT, GPT, ViT'de kullanılır")


def demonstrate_optimizer_comparison() -> None:
    """Optimizer'ları karşılaştırır"""
    print("\n" + "🎯 OPTIMIZER KARŞILAŞTIRMASI".center(70, "━"))
    
    # Basit problem: y = 3x + 2
    X = torch.randn(100, 1) * 10
    y = 3 * X + 2 + torch.randn(100, 1) * 0.5
    
    optimizers_config = {
        'SGD': lambda p: optim.SGD(p, lr=0.01),
        'SGD+Momentum': lambda p: optim.SGD(p, lr=0.01, momentum=0.9),
        'RMSprop': lambda p: optim.RMSprop(p, lr=0.01),
        'Adam': lambda p: optim.Adam(p, lr=0.01),
        'AdamW': lambda p: optim.AdamW(p, lr=0.01, weight_decay=0.01)
    }
    
    results = {}
    
    for name, opt_fn in optimizers_config.items():
        model = nn.Linear(1, 1)
        optimizer = opt_fn(model.parameters())
        
        # 100 epoch training
        for epoch in range(100):
            optimizer.zero_grad()
            pred = model(X)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        results[name] = final_loss
        
        print(f"{name:15} → Final loss: {final_loss:.6f}")
    
    best = min(results, key=results.get)
    print(f"\n🏆 En iyi: {best}")


def demonstrate_learning_rate_scheduling() -> None:
    """Learning rate scheduling"""
    print("\n" + "🎯 LEARNING RATE SCHEDULING".center(70, "━"))
    
    model = nn.Linear(2, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    # StepLR: Her 10 epoch'ta lr'yi 0.1 ile çarp
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    print(f"StepLR: Her 10 epoch'ta lr × 0.1")
    
    for epoch in range(30):
        # Dummy training
        optimizer.zero_grad()
        loss = torch.tensor(1.0, requires_grad=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}: lr = {current_lr:.6f}")


def main() -> None:
    print("\n" + "="*70)
    print("🚀 OPTIMIZER ALGORITHMS")
    print("="*70)
    
    demonstrate_sgd()
    demonstrate_momentum()
    demonstrate_rmsprop()
    demonstrate_adam()
    demonstrate_adamw()
    demonstrate_optimizer_comparison()
    demonstrate_learning_rate_scheduling()
    
    print("\n" + "="*70)
    print("✅ DERS 12 TAMAMLANDI!")
    print("="*70 + "\n")
    
    print("🎉 FAZ 2 TAMAMLANDI!")
    print("Sonraki: Faz 3 - Data Engineering")


if __name__ == "__main__":
    main()
