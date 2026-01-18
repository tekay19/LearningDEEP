"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DERS 06: AUTOGRAD ENGINE - DAG, BACKWARD VE GRADIENT FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Amaç: PyTorch'un otomatik türev mekanizmasını anlamak.
DAG (Directed Acyclic Graph) yapısını ve .backward() çalışma mantığını öğrenmek.

Hedef Kitle: Senior Developer'lar için "Under the Hood" analiz.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
import graphviz  # pip install graphviz


def demonstrate_basic_autograd() -> None:
    """
    Temel autograd mekanizmasını gösterir.
    """
    print("\n" + "🎯 BÖLÜM 1: TEMEL AUTOGRAD - OTOMATİK TÜREV".center(70, "━"))
    
    # requires_grad=True ile tensor oluşturma
    print("🔹 Gradient Takibi Aktif Tensor")
    
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    print(f"x = {x}")
    print(f"x.requires_grad = {x.requires_grad}")
    print(f"x.grad = {x.grad} (Henüz backward() çağrılmadı)\n")
    
    # Basit bir işlem
    print("─"*70)
    print("🔹 İşlem: y = x² + 3x + 1")
    
    y = x**2 + 3*x + 1
    print(f"y = {y}")
    print(f"y.requires_grad = {y.requires_grad}")
    print(f"y.grad_fn = {y.grad_fn}")
    print(f"💡 grad_fn: Bu tensor'u oluşturan işlem (AddBackward)\n")
    
    # Skaler çıktı için backward
    print("─"*70)
    print("🔹 Backward: Gradient Hesaplama")
    
    loss = y.sum()  # Skaler yapmalıyız
    print(f"loss = y.sum() = {loss}")
    print(f"loss.grad_fn = {loss.grad_fn}\n")
    
    loss.backward()
    print(f"✅ loss.backward() çağrıldı!")
    print(f"x.grad = {x.grad}")
    
    # Manuel doğrulama
    print(f"\n🧮 MANUEL DOĞRULAMA:")
    print(f"dy/dx = 2x + 3")
    print(f"x=2 için: 2(2) + 3 = 7")
    print(f"x=3 için: 2(3) + 3 = 9")
    print(f"PyTorch sonucu: {x.grad.tolist()}")
    print(f"✅ Eşleşiyor!")


def demonstrate_computational_graph() -> None:
    """
    Computational graph (hesaplama grafiği) yapısını gösterir.
    """
    print("\n" + "🎯 BÖLÜM 2: COMPUTATIONAL GRAPH - DAG YAPISI".center(70, "━"))
    
    # Daha karmaşık bir graph
    print("🔹 Karmaşık Hesaplama Grafiği")
    
    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([3.0], requires_grad=True)
    
    print(f"a = {a.item()}, b = {b.item()}\n")
    
    # İşlemler
    c = a * b           # c = 2 * 3 = 6
    d = a + b           # d = 2 + 3 = 5
    e = c * d           # e = 6 * 5 = 30
    f = e.relu()        # f = max(0, 30) = 30
    loss = f.sum()      # loss = 30
    
    print(f"c = a * b = {c.item()}")
    print(f"d = a + b = {d.item()}")
    print(f"e = c * d = {e.item()}")
    print(f"f = relu(e) = {f.item()}")
    print(f"loss = {loss.item()}\n")
    
    # Graph yapısını göster
    print("─"*70)
    print("🔹 Graph Yapısı (grad_fn chain)")
    
    print(f"loss.grad_fn = {loss.grad_fn}")
    print(f"  └─ f.grad_fn = {f.grad_fn}")
    print(f"      └─ e.grad_fn = {e.grad_fn}")
    print(f"          ├─ c.grad_fn = {c.grad_fn}")
    print(f"          └─ d.grad_fn = {d.grad_fn}\n")
    
    # Backward pass
    print("─"*70)
    print("🔹 Backward Pass: Gradient Akışı")
    
    loss.backward()
    
    print(f"∂loss/∂a = {a.grad.item()}")
    print(f"∂loss/∂b = {b.grad.item()}")
    
    # Manuel hesaplama
    print(f"\n🧮 MANUEL HESAPLAMA:")
    print(f"∂loss/∂e = ∂f/∂e = 1 (relu türevi, e>0 için)")
    print(f"∂loss/∂c = ∂loss/∂e × ∂e/∂c = 1 × d = {d.item()}")
    print(f"∂loss/∂d = ∂loss/∂e × ∂e/∂d = 1 × c = {c.item()}")
    print(f"∂loss/∂a = ∂loss/∂c × ∂c/∂a + ∂loss/∂d × ∂d/∂a")
    print(f"         = {d.item()} × {b.item()} + {c.item()} × 1 = {d.item() * b.item() + c.item()}")
    print(f"∂loss/∂b = ∂loss/∂c × ∂c/∂b + ∂loss/∂d × ∂d/∂b")
    print(f"         = {d.item()} × {a.item()} + {c.item()} × 1 = {d.item() * a.item() + c.item()}")


def demonstrate_gradient_accumulation() -> None:
    """
    Gradient accumulation (biriktirme) mekanizmasını gösterir.
    """
    print("\n" + "🎯 BÖLÜM 3: GRADIENT ACCUMULATION - BİRİKTİRME".center(70, "━"))
    
    # İlk backward
    print("🔹 İlk Backward")
    
    x = torch.tensor([2.0], requires_grad=True)
    
    y1 = x**2
    y1.backward()
    
    print(f"y1 = x² = {y1.item()}")
    print(f"∂y1/∂x = 2x = {x.grad.item()}\n")
    
    # İkinci backward (gradient birikiyor!)
    print("─"*70)
    print("🔴 İkinci Backward (Gradient Birikiyor!)")
    
    y2 = x**3
    y2.backward()
    
    print(f"y2 = x³ = {y2.item()}")
    print(f"x.grad = {x.grad.item()}")
    print(f"💡 Beklenen: ∂y2/∂x = 3x² = {3 * x.item()**2}")
    print(f"⚠️  Ama sonuç: {x.grad.item()} (Önceki gradient eklendi!)\n")
    
    # Gradient sıfırlama
    print("─"*70)
    print("✅ Gradient Sıfırlama")
    
    x.grad.zero_()  # veya x.grad = None
    
    y3 = x**3
    y3.backward()
    
    print(f"x.grad.zero_() çağrıldı")
    print(f"y3 = x³ backward sonrası:")
    print(f"x.grad = {x.grad.item()}")
    print(f"✅ Doğru sonuç: {3 * x.item()**2}\n")
    
    # Pratik: Training loop'ta kullanım
    print("─"*70)
    print("🔹 PRATİK: Training Loop'ta Kullanım")
    
    print("""
    # YANLIŞ
    for epoch in range(10):
        loss = model(x)
        loss.backward()  # Gradientler birikiyor!
        optimizer.step()
    
    # DOĞRU
    for epoch in range(10):
        optimizer.zero_grad()  # Gradientleri sıfırla
        loss = model(x)
        loss.backward()
        optimizer.step()
    """)


def demonstrate_no_grad_context() -> None:
    """
    torch.no_grad() ve torch.inference_mode() kullanımını gösterir.
    """
    print("\n" + "🎯 BÖLÜM 4: NO_GRAD VE INFERENCE_MODE".center(70, "━"))
    
    # Normal mod (gradient tracking)
    print("🔹 Normal Mod (Gradient Tracking)")
    
    x = torch.randn(1000, 1000, requires_grad=True)
    
    import time
    start = time.time()
    y = x @ x
    z = y.sum()
    normal_time = time.time() - start
    
    print(f"y.requires_grad = {y.requires_grad}")
    print(f"y.grad_fn = {y.grad_fn}")
    print(f"Süre: {normal_time*1000:.4f} ms\n")
    
    # torch.no_grad() context
    print("─"*70)
    print("🔹 torch.no_grad() Context")
    
    with torch.no_grad():
        start = time.time()
        y = x @ x
        z = y.sum()
        no_grad_time = time.time() - start
        
        print(f"y.requires_grad = {y.requires_grad}")
        print(f"y.grad_fn = {y.grad_fn}")
        print(f"Süre: {no_grad_time*1000:.4f} ms")
        print(f"🚀 {normal_time/no_grad_time:.2f}x daha hızlı!\n")
    
    # torch.inference_mode() (PyTorch 1.9+)
    print("─"*70)
    print("🔹 torch.inference_mode() (Daha Hızlı)")
    
    with torch.inference_mode():
        start = time.time()
        y = x @ x
        z = y.sum()
        inference_time = time.time() - start
        
        print(f"y.requires_grad = {y.requires_grad}")
        print(f"Süre: {inference_time*1000:.4f} ms")
        print(f"🚀 {normal_time/inference_time:.2f}x daha hızlı!")
        print(f"💡 inference_mode, no_grad'dan daha agresif optimizasyon yapar\n")
    
    # Decorator kullanımı
    print("─"*70)
    print("🔹 Decorator Kullanımı")
    
    print("""
    @torch.no_grad()
    def evaluate(model, data):
        predictions = model(data)
        return predictions
    
    @torch.inference_mode()
    def predict(model, data):
        return model(data)  # Daha hızlı!
    """)


def demonstrate_retain_graph() -> None:
    """
    retain_graph parametresini açıklar.
    """
    print("\n" + "🎯 BÖLÜM 5: RETAIN_GRAPH - GRAPH'I KORUMA".center(70, "━"))
    
    x = torch.tensor([2.0], requires_grad=True)
    
    y = x**2
    z = y * 3
    
    print(f"y = x² = {y.item()}")
    print(f"z = y × 3 = {z.item()}\n")
    
    # İlk backward
    print("─"*70)
    print("🔹 İlk Backward (y)")
    
    y.backward(retain_graph=True)
    print(f"y.backward(retain_graph=True)")
    print(f"x.grad = {x.grad.item()}\n")
    
    # İkinci backward (aynı graph)
    print("─"*70)
    print("🔹 İkinci Backward (z)")
    
    x.grad.zero_()
    z.backward()
    print(f"z.backward()")
    print(f"x.grad = {x.grad.item()}")
    
    # retain_graph=False durumu
    print("\n" + "─"*70)
    print("🔴 retain_graph=False (Default)")
    
    x = torch.tensor([2.0], requires_grad=True)
    y = x**2
    
    y.backward()  # Graph silinir
    
    try:
        y.backward()  # HATA: Graph yok!
    except RuntimeError as e:
        print(f"❌ HATA: {e}")
        print(f"💡 Graph bir kez kullanıldıktan sonra silinir (bellek tasarrufu)")


def demonstrate_higher_order_gradients() -> None:
    """
    İkinci dereceden türevleri (Hessian) hesaplar.
    """
    print("\n" + "🎯 BÖLÜM 6: HIGHER-ORDER GRADIENTS - İKİNCİ TÜREV".center(70, "━"))
    
    # Birinci türev
    print("🔹 Birinci Türev")
    
    x = torch.tensor([2.0], requires_grad=True)
    y = x**3  # y = x³
    
    print(f"y = x³ = {y.item()}")
    
    # dy/dx
    grad_y = torch.autograd.grad(y, x, create_graph=True)[0]
    print(f"dy/dx = 3x² = {grad_y.item()}\n")
    
    # İkinci türev
    print("─"*70)
    print("🔹 İkinci Türev (Hessian)")
    
    # d²y/dx²
    grad2_y = torch.autograd.grad(grad_y, x)[0]
    print(f"d²y/dx² = 6x = {grad2_y.item()}")
    print(f"Manuel: 6 × {x.item()} = {6 * x.item()}")
    print(f"✅ Eşleşiyor!\n")
    
    # Pratik: Newton's Method
    print("─"*70)
    print("🔹 PRATİK: Newton's Method Optimizasyonu")
    
    print("""
    # Newton's Method: x_new = x - f'(x) / f''(x)
    
    x = torch.tensor([1.0], requires_grad=True)
    
    for i in range(10):
        y = (x - 2)**2  # Minimize edilecek fonksiyon
        
        grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
        grad2 = torch.autograd.grad(grad1, x)[0]
        
        x.data -= grad1 / grad2  # Newton update
    
    # x → 2.0'a yakınsar (minimum nokta)
    """)


def demonstrate_gradient_checkpointing() -> None:
    """
    Gradient checkpointing ile bellek optimizasyonu.
    """
    print("\n" + "🎯 BÖLÜM 7: GRADIENT CHECKPOINTING - BELLEK OPTİMİZASYONU".center(70, "━"))
    
    print("🔹 Normal Backward (Tüm Intermediate Değerler Saklanır)")
    
    print("""
    # Normal backward
    x = torch.randn(1000, 1000, requires_grad=True)
    
    y1 = x @ x
    y2 = y1 @ y1
    y3 = y2 @ y2
    loss = y3.sum()
    
    loss.backward()
    
    # Bellek: y1, y2, y3 hepsi saklanır (backward için gerekli)
    # Toplam: ~4 GB (büyük modellerde problem!)
    """)
    
    print("\n" + "─"*70)
    print("🔹 Gradient Checkpointing (Bellek Tasarrufu)")
    
    print("""
    from torch.utils.checkpoint import checkpoint
    
    def compute_block(x):
        y1 = x @ x
        y2 = y1 @ y1
        return y2
    
    x = torch.randn(1000, 1000, requires_grad=True)
    
    # Checkpointing kullan
    y = checkpoint(compute_block, x)
    loss = y.sum()
    loss.backward()
    
    # Bellek: Sadece checkpoint noktaları saklanır
    # Backward sırasında intermediate değerler yeniden hesaplanır
    # Trade-off: %50 bellek tasarrufu, %30 yavaşlama
    """)
    
    print(f"\n💡 Kullanım Alanı: Transformer'lar, çok derin CNN'ler")


def demonstrate_common_pitfalls() -> None:
    """
    Autograd kullanımında sık yapılan hataları gösterir.
    """
    print("\n" + "🎯 BÖLÜM 8: YAYGIN HATALAR VE ÇÖZÜMLER".center(70, "━"))
    
    # HATA 1: In-place işlem
    print("🔴 HATA 1: In-place İşlem Gradient Graph'ı Bozar")
    
    x = torch.tensor([2.0], requires_grad=True)
    y = x**2
    
    try:
        # YANLIŞ: In-place işlem
        y.add_(1.0)  # y += 1
        y.backward()
    except RuntimeError as e:
        print(f"❌ HATA: {e}")
        print(f"💡 In-place işlemler (_ile bitenler) gradient graph'ı bozar\n")
    
    # HATA 2: Leaf variable'a in-place işlem
    print("─"*70)
    print("🔴 HATA 2: Leaf Variable'a In-place İşlem")
    
    x = torch.tensor([2.0], requires_grad=True)
    
    try:
        x.add_(1.0)  # HATA!
    except RuntimeError as e:
        print(f"❌ HATA: {e}")
        print(f"💡 Leaf variable'lar (input) değiştirilemez\n")
    
    # HATA 3: Non-scalar backward
    print("─"*70)
    print("🔴 HATA 3: Non-scalar Tensor'da backward()")
    
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x**2
    
    try:
        y.backward()  # HATA: y skaler değil!
    except RuntimeError as e:
        print(f"❌ HATA: {e}")
        print(f"💡 ÇÖZÜM 1: .sum() ile skaler yap")
        
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x**2
        y.sum().backward()
        print(f"y.sum().backward() → x.grad = {x.grad}")
        
        print(f"\n💡 ÇÖZÜM 2: gradient parametresi ver")
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x**2
        y.backward(torch.ones_like(y))
        print(f"y.backward(torch.ones_like(y)) → x.grad = {x.grad}")


def main() -> None:
    """
    Ana çalıştırma fonksiyonu.
    """
    print("\n" + "="*70)
    print("🚀 AUTOGRAD ENGINE - OTOMATİK TÜREV MEKANİZMASI".center(70))
    print("="*70)
    
    demonstrate_basic_autograd()
    demonstrate_computational_graph()
    demonstrate_gradient_accumulation()
    demonstrate_no_grad_context()
    demonstrate_retain_graph()
    demonstrate_higher_order_gradients()
    demonstrate_gradient_checkpointing()
    demonstrate_common_pitfalls()
    
    print("\n" + "="*70)
    print("✅ DERS 06 TAMAMLANDI!".center(70))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
