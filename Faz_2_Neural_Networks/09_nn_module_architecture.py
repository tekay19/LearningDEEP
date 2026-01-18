"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DERS 09: nn.Module MİMARİSİ - __init__ VE forward MEKANİZMASI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Amaç: nn.Module sınıf yapısını anlamak.
Parameter yönetimi, forward/backward mekanizması, model inspection.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from collections import OrderedDict


class SimpleLinear(nn.Module):
    """
    En basit nn.Module örneği: y = wx + b
    """
    
    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features: Giriş boyutu
            out_features: Çıkış boyutu
        """
        # ZORUNLU: super().__init__() çağrısı
        super().__init__()
        
        # Parametreleri tanımla
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        print(f"✅ SimpleLinear oluşturuldu: ({in_features} → {out_features})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = x @ W^T + b
        
        Args:
            x: (batch, in_features)
            
        Returns:
            y: (batch, out_features)
        """
        return x @ self.weight.t() + self.bias


def demonstrate_basic_module() -> None:
    """
    Basit nn.Module kullanımını gösterir.
    """
    print("\n" + "🎯 BÖLÜM 1: TEMEL nn.Module KULLANIMI".center(70, "━"))
    
    # Model oluştur
    model = SimpleLinear(3, 2)
    
    # Input
    x = torch.randn(5, 3)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    y = model(x)  # model.forward(x) ile aynı
    print(f"Output shape: {y.shape}")
    
    # Parametreleri incele
    print(f"\n📊 Parametreler:")
    for name, param in model.named_parameters():
        print(f"   {name}: {param.shape}, requires_grad={param.requires_grad}")
    
    # Toplam parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n   Toplam parametre: {total_params}")


class MultiLayerPerceptron(nn.Module):
    """
    Çok katmanlı perceptron (MLP).
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        """
        Args:
            input_size: Giriş boyutu
            hidden_sizes: Hidden layer boyutları [64, 32, ...]
            output_size: Çıkış boyutu
        """
        super().__init__()
        
        # Katmanları oluştur
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        # nn.Sequential ile birleştir
        self.network = nn.Sequential(*layers)
        
        print(f"✅ MLP oluşturuldu: {input_size} → {hidden_sizes} → {output_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        return self.network(x)


def demonstrate_sequential() -> None:
    """
    nn.Sequential kullanımını gösterir.
    """
    print("\n" + "🎯 BÖLÜM 2: nn.Sequential - KATMAN ZİNCİRİ".center(70, "━"))
    
    # Yöntem 1: Liste ile
    model1 = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Sigmoid()
    )
    
    print(f"\n📦 Sequential Model (liste):")
    print(model1)
    
    # Yöntem 2: OrderedDict ile (isimlendirme)
    model2 = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(10, 20)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(20, 10)),
        ('sigmoid', nn.Sigmoid())
    ]))
    
    print(f"\n📦 Sequential Model (OrderedDict):")
    print(model2)
    
    # Katmanlara erişim
    print(f"\n🔍 Katman Erişimi:")
    print(f"   model2.fc1: {model2.fc1}")
    print(f"   model2[0]: {model2[0]}")  # Index ile


class CustomModel(nn.Module):
    """
    Özel forward logic'li model.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor, use_skip: bool = True) -> torch.Tensor:
        """
        Forward pass with optional skip connection.
        
        Args:
            x: Input tensor
            use_skip: Skip connection kullan mı?
        """
        # İlk katman
        out = self.relu(self.fc1(x))
        
        # İkinci katman (skip connection ile)
        identity = out
        out = self.relu(self.fc2(out))
        
        if use_skip:
            out = out + identity  # Residual connection
        
        # Çıkış katmanı
        out = self.fc3(out)
        
        return out


def demonstrate_custom_forward() -> None:
    """
    Özel forward logic gösterir.
    """
    print("\n" + "🎯 BÖLÜM 3: ÖZEL FORWARD LOGIC".center(70, "━"))
    
    model = CustomModel(10, 20, 5)
    x = torch.randn(3, 10)
    
    # Skip connection ile
    y_with_skip = model(x, use_skip=True)
    print(f"\nSkip connection ile: {y_with_skip.shape}")
    
    # Skip connection olmadan
    y_without_skip = model(x, use_skip=False)
    print(f"Skip connection olmadan: {y_without_skip.shape}")
    
    # Fark
    diff = (y_with_skip - y_without_skip).abs().mean()
    print(f"Ortalama fark: {diff.item():.4f}")


def demonstrate_parameter_management() -> None:
    """
    Parameter yönetimini gösterir.
    """
    print("\n" + "🎯 BÖLÜM 4: PARAMETER YÖNETİMİ".center(70, "━"))
    
    model = MultiLayerPerceptron(10, [20, 15], 5)
    
    # Tüm parametreler
    print(f"\n📊 Tüm Parametreler:")
    for name, param in model.named_parameters():
        print(f"   {name}: {param.shape}")
    
    # Toplam parametre sayısı
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n   Toplam: {total:,}")
    print(f"   Trainable: {trainable:,}")
    
    # Belirli parametreleri dondur
    print(f"\n❄️  İlk katmanı dondurma:")
    for name, param in model.named_parameters():
        if 'network.0' in name:  # İlk Linear layer
            param.requires_grad = False
            print(f"   {name} donduruldu")
    
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n   Trainable (sonra): {trainable_after:,}")


def demonstrate_state_dict() -> None:
    """
    state_dict kullanımını gösterir.
    """
    print("\n" + "🎯 BÖLÜM 5: STATE_DICT - MODEL KAYDETME".center(70, "━"))
    
    # Model oluştur
    model = SimpleLinear(3, 2)
    
    # state_dict al
    state = model.state_dict()
    
    print(f"\n📦 state_dict içeriği:")
    for key, value in state.items():
        print(f"   {key}: {value.shape}")
    
    # Model kaydet
    torch.save(state, '/tmp/model.pth')
    print(f"\n💾 Model kaydedildi: /tmp/model.pth")
    
    # Yeni model oluştur ve yükle
    new_model = SimpleLinear(3, 2)
    new_model.load_state_dict(torch.load('/tmp/model.pth'))
    
    print(f"✅ Model yüklendi")
    
    # Parametreleri karşılaştır
    print(f"\n🔍 Parametre Karşılaştırması:")
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), new_model.named_parameters()):
        diff = (p1 - p2).abs().max()
        print(f"   {n1}: max diff = {diff.item():.2e}")


def demonstrate_train_eval_mode() -> None:
    """
    train() ve eval() modlarını gösterir.
    """
    print("\n" + "🎯 BÖLÜM 6: TRAIN VS EVAL MODU".center(70, "━"))
    
    # Dropout ve BatchNorm içeren model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(20, 5)
    )
    
    x = torch.randn(3, 10)
    
    # Training mode
    model.train()
    print(f"\n🏋️  Training Mode:")
    print(f"   model.training: {model.training}")
    
    y1 = model(x)
    y2 = model(x)
    diff_train = (y1 - y2).abs().mean()
    print(f"   İki forward pass farkı: {diff_train.item():.4f}")
    print(f"   💡 Dropout aktif, her seferinde farklı!")
    
    # Evaluation mode
    model.eval()
    print(f"\n🎯 Evaluation Mode:")
    print(f"   model.training: {model.training}")
    
    y1 = model(x)
    y2 = model(x)
    diff_eval = (y1 - y2).abs().mean()
    print(f"   İki forward pass farkı: {diff_eval.item():.4f}")
    print(f"   💡 Dropout kapalı, deterministik!")


def demonstrate_hooks() -> None:
    """
    Forward ve backward hooks gösterir.
    """
    print("\n" + "🎯 BÖLÜM 7: HOOKS - İÇ KATMAN ERİŞİMİ".center(70, "━"))
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Activation'ları sakla
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Hook kaydet
    model[0].register_forward_hook(get_activation('fc1'))
    model[2].register_forward_hook(get_activation('fc2'))
    
    # Forward pass
    x = torch.randn(3, 10)
    y = model(x)
    
    print(f"\n📊 Kaydedilen Activations:")
    for name, act in activations.items():
        print(f"   {name}: {act.shape}, mean={act.mean().item():.4f}")


def main() -> None:
    """
    Ana çalıştırma fonksiyonu.
    """
    print("\n" + "="*70)
    print("🚀 nn.Module MİMARİSİ")
    print("="*70)
    
    demonstrate_basic_module()
    demonstrate_sequential()
    demonstrate_custom_forward()
    demonstrate_parameter_management()
    demonstrate_state_dict()
    demonstrate_train_eval_mode()
    demonstrate_hooks()
    
    print("\n" + "="*70)
    print("✅ DERS 09 TAMAMLANDI!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
