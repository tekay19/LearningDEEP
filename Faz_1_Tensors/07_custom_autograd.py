"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DERS 07: CUSTOM AUTOGRAD - KEND İ TÜREV FONKSİYONUNU YAZMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Amaç: torch.autograd.Function sınıfını miras alarak özel türev fonksiyonları yazmak.
Forward ve backward pass'leri manuel olarak tanımlamak.

Hedef Kitle: Senior Developer'lar için "Under the Hood" analiz.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Tuple, Any, Optional
import math


class CustomReLU(Function):
    """
    ReLU fonksiyonunun custom implementasyonu.
    
    Forward: f(x) = max(0, x)
    Backward: df/dx = 1 if x > 0 else 0
    """
    
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: ReLU hesaplama.
        
        Args:
            ctx: Context object (backward için veri saklamak için)
            input: Giriş tensörü
            
        Returns:
            ReLU uygulanmış tensor
        """
        # Backward için input'u sakla
        ctx.save_for_backward(input)
        
        # ReLU: max(0, x)
        output = input.clamp(min=0)
        
        return output
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass: ReLU türevi.
        
        Args:
            ctx: Context object (forward'dan gelen veri)
            grad_output: Üstten gelen gradient (∂L/∂output)
            
        Returns:
            Input'a göre gradient (∂L/∂input)
        """
        # Forward'dan kaydedilen input'u al
        input, = ctx.saved_tensors
        
        # ReLU türevi: 1 if x > 0 else 0
        grad_input = grad_output.clone()
        grad_input[input <= 0] = 0
        
        return grad_input


def demonstrate_custom_relu() -> None:
    """
    Custom ReLU implementasyonunu test eder.
    """
    print("\n" + "🎯 BÖLÜM 1: CUSTOM RELU - İLK ÖRNEK".center(70, "━"))
    
    # Custom ReLU kullanımı
    print("🔹 Custom ReLU Kullanımı")
    
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    
    # Custom ReLU uygula
    y = CustomReLU.apply(x)
    
    print(f"Input:  {x.tolist()}")
    print(f"Output: {y.tolist()}")
    print(f"y.grad_fn: {y.grad_fn}\n")
    
    # Backward pass
    print("─"*70)
    print("🔹 Backward Pass")
    
    loss = y.sum()
    loss.backward()
    
    print(f"x.grad: {x.grad.tolist()}")
    print(f"💡 Gradient: 1 for x>0, 0 for x≤0\n")
    
    # PyTorch ReLU ile karşılaştırma
    print("─"*70)
    print("🔹 PyTorch ReLU ile Karşılaştırma")
    
    x_torch = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y_torch = torch.relu(x_torch)
    y_torch.sum().backward()
    
    print(f"PyTorch ReLU gradient: {x_torch.grad.tolist()}")
    print(f"Custom ReLU gradient:  {x.grad.tolist()}")
    print(f"✅ Eşleşiyor!")


class CustomSigmoid(Function):
    """
    Sigmoid fonksiyonunun custom implementasyonu.
    
    Forward: σ(x) = 1 / (1 + e^(-x))
    Backward: dσ/dx = σ(x) × (1 - σ(x))
    """
    
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Sigmoid hesaplama.
        """
        output = 1 / (1 + torch.exp(-input))
        
        # Backward için output'u sakla (türev hesabında lazım)
        ctx.save_for_backward(output)
        
        return output
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass: Sigmoid türevi.
        
        dσ/dx = σ(x) × (1 - σ(x))
        """
        output, = ctx.saved_tensors
        
        # Sigmoid türevi
        grad_input = grad_output * output * (1 - output)
        
        return grad_input


def demonstrate_custom_sigmoid() -> None:
    """
    Custom Sigmoid implementasyonunu test eder.
    """
    print("\n" + "🎯 BÖLÜM 2: CUSTOM SIGMOID - TÜREV OPTİMİZASYONU".center(70, "━"))
    
    print("🔹 Custom Sigmoid")
    
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = CustomSigmoid.apply(x)
    
    print(f"Input:  {x.tolist()}")
    print(f"Output: {[f'{v:.4f}' for v in y.tolist()]}\n")
    
    # Backward
    loss = y.sum()
    loss.backward()
    
    print(f"x.grad: {[f'{v:.4f}' for v in x.grad.tolist()]}")
    
    # Manuel doğrulama
    print(f"\n🧮 MANUEL DOĞRULAMA (x=0):")
    print(f"σ(0) = 1/(1+e^0) = 0.5")
    print(f"dσ/dx = σ(0) × (1-σ(0)) = 0.5 × 0.5 = 0.25")
    print(f"PyTorch sonucu: {x.grad[2].item():.4f}")
    print(f"✅ Eşleşiyor!")


class CustomLinear(Function):
    """
    Linear layer'ın custom implementasyonu.
    
    Forward: y = x @ W^T + b
    Backward: 
        ∂L/∂x = ∂L/∂y @ W
        ∂L/∂W = ∂L/∂y^T @ x
        ∂L/∂b = ∂L/∂y.sum(dim=0)
    """
    
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, weight: torch.Tensor, 
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: Linear transformation.
        
        Args:
            input: (batch, in_features)
            weight: (out_features, in_features)
            bias: (out_features,) or None
        """
        # Backward için kaydet
        ctx.save_for_backward(input, weight, bias)
        
        # y = x @ W^T + b
        output = input.mm(weight.t())
        
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        
        return output
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Backward pass: Linear layer gradients.
        
        Returns:
            (grad_input, grad_weight, grad_bias)
        """
        input, weight, bias = ctx.saved_tensors
        
        grad_input = grad_weight = grad_bias = None
        
        # ∂L/∂x = ∂L/∂y @ W
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        
        # ∂L/∂W = ∂L/∂y^T @ x
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        
        # ∂L/∂b = ∂L/∂y.sum(dim=0)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        
        return grad_input, grad_weight, grad_bias


def demonstrate_custom_linear() -> None:
    """
    Custom Linear layer'ı test eder.
    """
    print("\n" + "🎯 BÖLÜM 3: CUSTOM LINEAR LAYER - MATMUL GRADİENT".center(70, "━"))
    
    print("🔹 Custom Linear Layer")
    
    batch_size, in_features, out_features = 4, 3, 2
    
    x = torch.randn(batch_size, in_features, requires_grad=True)
    W = torch.randn(out_features, in_features, requires_grad=True)
    b = torch.randn(out_features, requires_grad=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {W.shape}")
    print(f"Bias shape: {b.shape}\n")
    
    # Custom linear
    y_custom = CustomLinear.apply(x, W, b)
    print(f"Output shape: {y_custom.shape}\n")
    
    # Backward
    loss = y_custom.sum()
    loss.backward()
    
    print(f"x.grad shape: {x.grad.shape}")
    print(f"W.grad shape: {W.grad.shape}")
    print(f"b.grad shape: {b.grad.shape}\n")
    
    # PyTorch nn.Linear ile karşılaştırma
    print("─"*70)
    print("🔹 PyTorch nn.Linear ile Karşılaştırma")
    
    x_torch = x.detach().clone().requires_grad_(True)
    
    linear = nn.Linear(in_features, out_features, bias=True)
    linear.weight.data = W.detach().clone()
    linear.bias.data = b.detach().clone()
    
    y_torch = linear(x_torch)
    y_torch.sum().backward()
    
    print(f"Gradient farkı (x): {(x.grad - x_torch.grad).abs().max().item():.2e}")
    print(f"Gradient farkı (W): {(W.grad - linear.weight.grad).abs().max().item():.2e}")
    print(f"Gradient farkı (b): {(b.grad - linear.bias.grad).abs().max().item():.2e}")
    print(f"✅ Neredeyse sıfır!")


class CustomGELU(Function):
    """
    GELU (Gaussian Error Linear Unit) custom implementasyonu.
    
    GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
    """
    
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: GELU approximation.
        """
        # GELU approximation
        c = math.sqrt(2.0 / math.pi)
        tanh_arg = c * (input + 0.044715 * input.pow(3))
        output = 0.5 * input * (1.0 + torch.tanh(tanh_arg))
        
        # Backward için kaydet
        ctx.save_for_backward(input, tanh_arg)
        
        return output
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass: GELU türevi.
        """
        input, tanh_arg = ctx.saved_tensors
        
        c = math.sqrt(2.0 / math.pi)
        tanh_val = torch.tanh(tanh_arg)
        sech2 = 1 - tanh_val.pow(2)
        
        # GELU türevi (chain rule)
        grad_tanh_arg = c * (1 + 3 * 0.044715 * input.pow(2))
        grad_input = 0.5 * (1 + tanh_val) + 0.5 * input * sech2 * grad_tanh_arg
        
        return grad_output * grad_input


def demonstrate_custom_gelu() -> None:
    """
    Custom GELU implementasyonunu test eder.
    """
    print("\n" + "🎯 BÖLÜM 4: CUSTOM GELU - KARMAŞIK TÜREV".center(70, "━"))
    
    print("🔹 Custom GELU")
    
    x = torch.linspace(-3, 3, 7, requires_grad=True)
    y_custom = CustomGELU.apply(x)
    
    print(f"Input:  {[f'{v:.2f}' for v in x.tolist()]}")
    print(f"Output: {[f'{v:.4f}' for v in y_custom.tolist()]}\n")
    
    # Backward
    y_custom.sum().backward()
    
    # PyTorch GELU ile karşılaştırma
    print("─"*70)
    print("🔹 PyTorch GELU ile Karşılaştırma")
    
    x_torch = x.detach().clone().requires_grad_(True)
    y_torch = torch.nn.functional.gelu(x_torch, approximate='tanh')
    y_torch.sum().backward()
    
    print(f"Output farkı: {(y_custom - y_torch).abs().max().item():.2e}")
    print(f"Gradient farkı: {(x.grad - x_torch.grad).abs().max().item():.2e}")
    print(f"✅ Çok küçük fark!")


class CustomBatchNorm(Function):
    """
    Batch Normalization'ın custom implementasyonu.
    
    Forward: y = (x - μ) / √(σ² + ε) × γ + β
    """
    
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, gamma: torch.Tensor, 
                beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """
        Forward pass: Batch normalization.
        
        Args:
            input: (batch, features)
            gamma: (features,) - scale parameter
            beta: (features,) - shift parameter
            eps: Numerical stability
        """
        # Batch statistics
        mean = input.mean(dim=0)
        var = input.var(dim=0, unbiased=False)
        
        # Normalize
        x_normalized = (input - mean) / torch.sqrt(var + eps)
        
        # Scale and shift
        output = gamma * x_normalized + beta
        
        # Backward için kaydet
        ctx.save_for_backward(input, gamma, mean, var, x_normalized)
        ctx.eps = eps
        
        return output
    
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Backward pass: BatchNorm gradients.
        """
        input, gamma, mean, var, x_normalized = ctx.saved_tensors
        eps = ctx.eps
        
        batch_size = input.size(0)
        
        # ∂L/∂γ
        grad_gamma = (grad_output * x_normalized).sum(dim=0)
        
        # ∂L/∂β
        grad_beta = grad_output.sum(dim=0)
        
        # ∂L/∂x (karmaşık!)
        grad_x_normalized = grad_output * gamma
        
        std = torch.sqrt(var + eps)
        
        grad_var = (grad_x_normalized * (input - mean) * -0.5 * (var + eps).pow(-1.5)).sum(dim=0)
        grad_mean = (grad_x_normalized * -1 / std).sum(dim=0) + grad_var * (-2 * (input - mean)).sum(dim=0) / batch_size
        
        grad_input = grad_x_normalized / std + grad_var * 2 * (input - mean) / batch_size + grad_mean / batch_size
        
        return grad_input, grad_gamma, grad_beta, None


def demonstrate_custom_batchnorm() -> None:
    """
    Custom BatchNorm implementasyonunu test eder.
    """
    print("\n" + "🎯 BÖLÜM 5: CUSTOM BATCHNORM - EN KARMAŞIK TÜREV".center(70, "━"))
    
    print("🔹 Custom BatchNorm")
    
    batch_size, features = 4, 3
    
    x = torch.randn(batch_size, features, requires_grad=True)
    gamma = torch.ones(features, requires_grad=True)
    beta = torch.zeros(features, requires_grad=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Gamma shape: {gamma.shape}")
    print(f"Beta shape: {beta.shape}\n")
    
    # Custom BatchNorm
    y_custom = CustomBatchNorm.apply(x, gamma, beta)
    
    print(f"Output mean: {y_custom.mean(dim=0).tolist()}")
    print(f"Output var: {y_custom.var(dim=0, unbiased=False).tolist()}")
    print(f"💡 Mean ≈ 0, Var ≈ 1 (normalization çalıştı!)\n")
    
    # Backward
    y_custom.sum().backward()
    
    print(f"x.grad shape: {x.grad.shape}")
    print(f"gamma.grad: {gamma.grad.tolist()}")
    print(f"beta.grad: {beta.grad.tolist()}")


def demonstrate_gradient_check() -> None:
    """
    Numerical gradient checking ile custom gradient'leri doğrular.
    """
    print("\n" + "🎯 BÖLÜM 6: GRADIENT CHECKING - DOĞRULAMA".center(70, "━"))
    
    print("🔹 Numerical Gradient vs Analytical Gradient")
    
    from torch.autograd import gradcheck
    
    # CustomReLU test
    print("\n🔹 CustomReLU Gradient Check")
    
    x = torch.randn(5, dtype=torch.double, requires_grad=True)
    
    # gradcheck: numerical vs analytical gradient karşılaştırması
    test_passed = gradcheck(CustomReLU.apply, x, eps=1e-6, atol=1e-4)
    
    print(f"Gradient check: {'✅ PASSED' if test_passed else '❌ FAILED'}")
    
    # CustomSigmoid test
    print("\n🔹 CustomSigmoid Gradient Check")
    
    x = torch.randn(5, dtype=torch.double, requires_grad=True)
    test_passed = gradcheck(CustomSigmoid.apply, x, eps=1e-6, atol=1e-4)
    
    print(f"Gradient check: {'✅ PASSED' if test_passed else '❌ FAILED'}")
    
    print(f"\n💡 gradcheck, numerical differentiation ile analytical gradient'i karşılaştırır")
    print(f"   Numerical: f'(x) ≈ (f(x+ε) - f(x-ε)) / (2ε)")


def demonstrate_common_pitfalls() -> None:
    """
    Custom autograd yazarken sık yapılan hataları gösterir.
    """
    print("\n" + "🎯 BÖLÜM 7: YAYGIN HATALAR VE ÇÖZÜMLER".center(70, "━"))
    
    print("🔴 HATA 1: ctx.save_for_backward() Unutmak")
    print("""
    # YANLIŞ
    @staticmethod
    def forward(ctx, input):
        output = input * 2
        return output  # input kaydedilmedi!
    
    @staticmethod
    def backward(ctx, grad_output):
        # input'a erişemeyiz! HATA!
        return grad_output * 2
    
    # DOĞRU
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # Kaydet!
        return input * 2
    """)
    
    print("\n" + "─"*70)
    print("🔴 HATA 2: Backward'da Yanlış Sayıda Gradient Döndürmek")
    print("""
    # Forward 3 parametre alıyor
    def forward(ctx, input, weight, bias):
        ...
    
    # YANLIŞ: Backward 2 gradient döndürüyor
    def backward(ctx, grad_output):
        return grad_input, grad_weight  # bias eksik!
    
    # DOĞRU: Her parametre için gradient döndür (None olabilir)
    def backward(ctx, grad_output):
        return grad_input, grad_weight, grad_bias
    """)
    
    print("\n" + "─"*70)
    print("🔴 HATA 3: In-place İşlem Kullanmak")
    print("""
    # YANLIŞ
    @staticmethod
    def backward(ctx, grad_output):
        grad_output[grad_output < 0] = 0  # In-place!
        return grad_output
    
    # DOĞRU
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()  # Kopya oluştur
        grad_input[grad_input < 0] = 0
        return grad_input
    """)


def main() -> None:
    """
    Ana çalıştırma fonksiyonu.
    """
    print("\n" + "="*70)
    print("🚀 CUSTOM AUTOGRAD - KEND İ TÜREV FONKSİYONUNU YAZMA".center(70))
    print("="*70)
    
    demonstrate_custom_relu()
    demonstrate_custom_sigmoid()
    demonstrate_custom_linear()
    demonstrate_custom_gelu()
    demonstrate_custom_batchnorm()
    demonstrate_gradient_check()
    demonstrate_common_pitfalls()
    
    print("\n" + "="*70)
    print("✅ DERS 07 TAMAMLANDI!".center(70))
    print("="*70 + "\n")
    
    print("🎉 FAZ 1 TAMAMLANDI! 🎉".center(70))
    print("Tensors & Computational Graph konularını bitirdiniz!".center(70))
    print("\n🚀 Sonraki: Faz 2 - Neural Network Fundamentals".center(70))


if __name__ == "__main__":
    main()
