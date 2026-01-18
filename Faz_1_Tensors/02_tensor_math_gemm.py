"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DERS 02: TENSOR MATEMATİĞİ - GEMM VE BROADCASTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Amaç: GEMM (General Matrix Multiply) algoritmasını anlamak.
Broadcasting kurallarını öğrenmek ve Vectorization avantajlarını görmek.

Hedef Kitle: Senior Developer'lar için "Under the Hood" analiz.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import numpy as np
import time
from typing import Tuple, List
import matplotlib.pyplot as plt


def inspect_operation(name: str, tensor: torch.Tensor, operation: str = "") -> None:
    """
    Bir tensor işleminin detaylarını yazdırır.
    
    Args:
        name: İşlem adı
        tensor: Sonuç tensor
        operation: İşlem açıklaması
    """
    print(f"\n{'─'*70}")
    print(f"🔬 {name}")
    if operation:
        print(f"📝 İşlem: {operation}")
    print(f"{'─'*70}")
    print(f"Shape: {tensor.shape}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Requires Grad: {tensor.requires_grad}")
    print(f"Data:\n{tensor}")
    print(f"{'─'*70}")


def demonstrate_matrix_multiplication_types() -> None:
    """
    Farklı çarpma türlerini (element-wise, dot, matmul) karşılaştırır.
    """
    print("\n" + "🎯 BÖLÜM 1: ÇARPMA TÜRLERİ - ELEMENT-WISE VS DOT VS MATMUL".center(70, "━"))
    
    # 1D Tensor'lar
    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    b = torch.tensor([4, 5, 6], dtype=torch.float32)
    
    print(f"\n📊 Vektörler:")
    print(f"a = {a}")
    print(f"b = {b}")
    
    # Element-wise çarpma (Hadamard product)
    element_wise = a * b
    inspect_operation(
        "Element-wise Çarpma (a * b)", 
        element_wise,
        "[1*4, 2*5, 3*6] = [4, 10, 18]"
    )
    
    # Dot product (İç çarpım)
    dot = torch.dot(a, b)
    inspect_operation(
        "Dot Product (torch.dot)", 
        dot,
        "1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32"
    )
    
    # @ operatörü (matmul için)
    dot_operator = a @ b
    inspect_operation(
        "@ Operatörü (a @ b)", 
        dot_operator,
        "1D tensor'larda dot product ile aynı"
    )
    
    # 2D Matrisler
    print("\n" + "─"*70)
    print("📊 Matrisler:")
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    
    print(f"\nA (2x2):\n{A}")
    print(f"\nB (2x2):\n{B}")
    
    # Element-wise çarpma
    element_wise_2d = A * B
    inspect_operation(
        "Element-wise Çarpma (A * B)", 
        element_wise_2d,
        "Her eleman kendi karşılığıyla çarpılır"
    )
    
    # Matrix multiplication (GEMM)
    matmul = A @ B
    inspect_operation(
        "Matrix Multiplication (A @ B)", 
        matmul,
        "C[i,j] = Σ(A[i,k] * B[k,j])"
    )
    
    # Manuel hesaplama doğrulaması
    print("\n🔍 MANUEL DOĞRULAMA:")
    print(f"C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = {A[0,0]}*{B[0,0]} + {A[0,1]}*{B[1,0]} = {A[0,0]*B[0,0] + A[0,1]*B[1,0]}")
    print(f"C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = {A[0,0]}*{B[0,1]} + {A[0,1]}*{B[1,1]} = {A[0,0]*B[0,1] + A[0,1]*B[1,1]}")
    print(f"Sonuç matrisi:\n{matmul}")


def demonstrate_gemm_performance() -> None:
    """
    GEMM optimizasyonlarını ve performans farklarını gösterir.
    """
    print("\n" + "🎯 BÖLÜM 2: GEMM PERFORMANSI - NAIVE VS OPTIMIZED".center(70, "━"))
    
    # Naive implementasyon (3 nested loop)
    def naive_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Üç iç içe döngü ile matris çarpımı (Eğitim amaçlı).
        ASLA production'da kullanma!
        """
        m, k = A.shape
        k2, n = B.shape
        assert k == k2, "İç boyutlar eşleşmeli!"
        
        C = torch.zeros(m, n, dtype=A.dtype)
        
        # O(m * n * k) karmaşıklık
        for i in range(m):
            for j in range(n):
                for p in range(k):
                    C[i, j] += A[i, p] * B[p, j]
        
        return C
    
    # Test matrisleri
    size = 128
    A = torch.randn(size, size)
    B = torch.randn(size, size)
    
    print(f"\n📊 Test Matrisleri: {size}x{size}")
    
    # Naive implementasyon
    start = time.time()
    C_naive = naive_matmul(A, B)
    naive_time = time.time() - start
    print(f"\n⏱️  Naive (3 Loop): {naive_time:.4f} saniye")
    
    # PyTorch optimized GEMM
    start = time.time()
    C_optimized = A @ B
    optimized_time = time.time() - start
    print(f"⏱️  PyTorch GEMM: {optimized_time:.6f} saniye")
    
    # Hız farkı
    speedup = naive_time / optimized_time
    print(f"\n🚀 HIZ ARTIŞI: {speedup:.0f}x daha hızlı!")
    print(f"💡 Sebep: BLAS/cuBLAS kütüphaneleri (C++/CUDA optimizasyonu)")
    
    # Sonuç doğrulaması
    diff = torch.abs(C_naive - C_optimized).max()
    print(f"\n✅ Sonuç Doğrulaması: Max fark = {diff:.2e} (Neredeyse sıfır)")


def demonstrate_broadcasting_rules() -> None:
    """
    PyTorch broadcasting kurallarını detaylı açıklar.
    """
    print("\n" + "🎯 BÖLÜM 3: BROADCASTING - OTOMATİK BOYUT GENİŞLETME".center(70, "━"))
    
    print("\n📜 BROADCASTING KURALLARI:")
    print("1. Sağdan sola doğru boyutları karşılaştır")
    print("2. İki boyut eşit VEYA birisi 1 ise uyumlu")
    print("3. Eksik boyutlar 1 kabul edilir")
    print("4. Uyumsuz boyutlar hata verir\n")
    
    # Örnek 1: Skaler ile tensor
    print("─"*70)
    print("📊 ÖRNEK 1: Skaler + Tensor")
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    scalar = 10.0
    
    result = tensor + scalar
    inspect_operation(
        "Skaler Broadcasting",
        result,
        f"{tensor.shape} + () → {scalar} tüm elemanlara eklenir"
    )
    
    # Örnek 2: 1D + 2D
    print("\n" + "─"*70)
    print("📊 ÖRNEK 2: 1D Tensor + 2D Tensor")
    matrix = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)  # (2, 3)
    vector = torch.tensor([10, 20, 30], dtype=torch.float32)  # (3,)
    
    result = matrix + vector
    print(f"\nMatrix shape: {matrix.shape}")
    print(f"Vector shape: {vector.shape}")
    print(f"Result shape: {result.shape}")
    print(f"\nMatrix:\n{matrix}")
    print(f"\nVector: {vector}")
    print(f"\nResult (her satıra vector eklendi):\n{result}")
    
    # Örnek 3: Farklı boyutlar
    print("\n" + "─"*70)
    print("📊 ÖRNEK 3: Karmaşık Broadcasting")
    a = torch.randn(3, 1, 5)  # (3, 1, 5)
    b = torch.randn(1, 4, 5)  # (1, 4, 5)
    
    result = a + b
    print(f"\na shape: {a.shape}")
    print(f"b shape: {b.shape}")
    print(f"Result shape: {result.shape}")
    print(f"\n💡 Açıklama:")
    print(f"  Dim 0: 3 vs 1 → 3 (b genişler)")
    print(f"  Dim 1: 1 vs 4 → 4 (a genişler)")
    print(f"  Dim 2: 5 vs 5 → 5 (eşit)")
    print(f"  Sonuç: (3, 4, 5)")
    
    # HATA ÖRNEĞİ
    print("\n" + "─"*70)
    print("🔴 HATA ÖRNEĞİ: Uyumsuz Boyutlar")
    x = torch.randn(3, 4)
    y = torch.randn(5, 4)
    
    try:
        result = x + y
    except RuntimeError as e:
        print(f"❌ HATA: {e}")
        print(f"\n💡 Sebep:")
        print(f"  x: (3, 4)")
        print(f"  y: (5, 4)")
        print(f"  Dim 0: 3 vs 5 → Uyumsuz! (İkisi de 1 değil)")


def demonstrate_vectorization_advantage() -> None:
    """
    Vectorization'ın performans avantajını gösterir.
    """
    print("\n" + "🎯 BÖLÜM 4: VECTORIZATION - DÖNGÜSÜZ HESAPLAMA".center(70, "━"))
    
    # Problem: 1 milyon elemanlı iki vektörü topla
    size = 1_000_000
    a = torch.randn(size)
    b = torch.randn(size)
    
    print(f"\n📊 Problem: {size:,} elemanlı vektör toplama")
    
    # Yöntem 1: Python loop (KÖTÜ)
    start = time.time()
    result_loop = torch.zeros(size)
    for i in range(size):
        result_loop[i] = a[i] + b[i]
    loop_time = time.time() - start
    print(f"\n⏱️  Python Loop: {loop_time:.4f} saniye")
    
    # Yöntem 2: Vectorized (İYİ)
    start = time.time()
    result_vec = a + b
    vec_time = time.time() - start
    print(f"⏱️  Vectorized: {vec_time:.6f} saniye")
    
    # Hız farkı
    speedup = loop_time / vec_time
    print(f"\n🚀 HIZ ARTIŞI: {speedup:.0f}x daha hızlı!")
    
    print(f"\n💡 Sebep:")
    print(f"  - Vectorized işlemler CPU SIMD (Single Instruction Multiple Data) kullanır")
    print(f"  - Bir komutla 4-8-16 eleman aynı anda işlenir")
    print(f"  - Python loop'ta her iterasyon için interpreter overhead var")
    
    # Batch işlemler
    print("\n" + "─"*70)
    print("📊 BATCH İŞLEMLER (Broadcasting + Vectorization)")
    
    # 1000 vektörün her birine farklı skaler ekle
    vectors = torch.randn(1000, 512)  # (batch, features)
    scalars = torch.randn(1000, 1)    # (batch, 1)
    
    start = time.time()
    result = vectors + scalars  # Broadcasting!
    batch_time = time.time() - start
    
    print(f"\nVectors: {vectors.shape}")
    print(f"Scalars: {scalars.shape}")
    print(f"Result: {result.shape}")
    print(f"⏱️  Süre: {batch_time:.6f} saniye")
    print(f"💡 1000 işlem tek seferde yapıldı (SIMD + Broadcasting)")


def demonstrate_common_pitfalls() -> None:
    """
    Sık yapılan hataları ve çözümlerini gösterir.
    """
    print("\n" + "🎯 BÖLÜM 5: YAYGIN HATALAR VE ÇÖZÜMLER".center(70, "━"))
    
    # HATA 1: Yanlış boyut sırası
    print("\n🔴 HATA 1: Matmul Boyut Uyumsuzluğu")
    A = torch.randn(3, 4)
    B = torch.randn(3, 5)
    
    print(f"A: {A.shape}, B: {B.shape}")
    
    try:
        # YANLIŞ: İç boyutlar eşleşmiyor
        C = A @ B
    except RuntimeError as e:
        print(f"❌ HATA: {e}")
        print(f"\n💡 ÇÖZÜM: B'yi transpose et")
        B_correct = torch.randn(4, 5)
        C = A @ B_correct
        print(f"✅ A @ B_correct: {A.shape} @ {B_correct.shape} = {C.shape}")
    
    # HATA 2: In-place işlem broadcasting'de
    print("\n" + "─"*70)
    print("🔴 HATA 2: In-place Broadcasting Hatası")
    x = torch.randn(3, 4)
    y = torch.randn(4)
    
    try:
        # YANLIŞ: In-place işlem boyut değiştiremez
        x += y  # Bu çalışır çünkü result shape = (3, 4)
        print(f"✅ x += y çalıştı: {x.shape}")
        
        # Ama tersi çalışmaz
        y_test = torch.randn(4)
        x_test = torch.randn(3, 4)
        # y_test += x_test  # Bu hata verir!
    except RuntimeError as e:
        print(f"❌ HATA: {e}")
    
    # HATA 3: Dtype uyumsuzluğu
    print("\n" + "─"*70)
    print("🔴 HATA 3: Dtype Karışıklığı")
    int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
    float_tensor = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float32)
    
    # PyTorch otomatik type promotion yapar
    result = int_tensor + float_tensor
    print(f"int32 + float32 = {result.dtype}")
    print(f"💡 PyTorch otomatik olarak float32'ye yükseltti")
    
    # Ama matmul'da dikkatli ol
    A_int = torch.randint(0, 10, (3, 4), dtype=torch.int32)
    B_int = torch.randint(0, 10, (4, 5), dtype=torch.int32)
    C_int = A_int @ B_int
    print(f"\nint32 @ int32 = {C_int.dtype}")
    print(f"⚠️  Overflow riski var! Büyük değerlerde float kullan")


def main() -> None:
    """
    Ana çalıştırma fonksiyonu.
    """
    print("\n" + "="*70)
    print("🚀 TENSOR MATEMATİĞİ - GEMM VE BROADCASTING".center(70))
    print("="*70)
    
    demonstrate_matrix_multiplication_types()
    demonstrate_gemm_performance()
    demonstrate_broadcasting_rules()
    demonstrate_vectorization_advantage()
    demonstrate_common_pitfalls()
    
    print("\n" + "="*70)
    print("✅ DERS 02 TAMAMLANDI!".center(70))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
