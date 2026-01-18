"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DERS 03: GELİŞMİŞ İNDEXLEME - MASKING, FANCY INDEXING VE VIEW VS COPY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Amaç: PyTorch'un gelişmiş indexleme tekniklerini öğrenmek.
Boolean masking, fancy indexing ve view/copy ayrımını anlamak.

Hedef Kitle: Senior Developer'lar için "Under the Hood" analiz.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
import time


def demonstrate_basic_indexing() -> None:
    """
    Temel indexleme türlerini gösterir: integer, slice, ellipsis.
    """
    print("\n" + "🎯 BÖLÜM 1: TEMEL İNDEXLEME - INTEGER, SLICE, ELLIPSIS".center(70, "━"))
    
    tensor = torch.arange(24).reshape(2, 3, 4)
    print(f"📊 Orijinal Tensor (2×3×4):\n{tensor}")
    print(f"Shape: {tensor.shape}\n")
    
    # Integer indexing
    print("─"*70)
    print("🔹 INTEGER INDEXING")
    element = tensor[0, 1, 2]
    print(f"tensor[0, 1, 2] = {element}")
    print(f"Shape: {element.shape} (0D tensor - scalar)")
    print(f"Storage paylaşımı: {tensor.data_ptr() == element.data_ptr()}")
    
    # Slice indexing
    print("\n" + "─"*70)
    print("🔹 SLICE INDEXING")
    sliced = tensor[0, :, 1:3]
    print(f"tensor[0, :, 1:3]:\n{sliced}")
    print(f"Shape: {sliced.shape}")
    print(f"Stride: {sliced.stride()}")
    print(f"Is contiguous: {sliced.is_contiguous()}")
    print(f"Storage paylaşımı: {tensor.data_ptr() == sliced.data_ptr()} (VIEW!)")
    
    # Ellipsis (...) kullanımı
    print("\n" + "─"*70)
    print("🔹 ELLIPSIS (...) KULLANIMI")
    print(f"tensor[..., 0] (Son boyutta 0. index):\n{tensor[..., 0]}")
    print(f"tensor[0, ...] (İlk boyutta 0. index):\n{tensor[0, ...]}")
    print(f"\n💡 Ellipsis = 'Geri kalan tüm boyutlar'")


def demonstrate_boolean_masking() -> None:
    """
    Boolean masking ile koşullu indexleme gösterir.
    """
    print("\n" + "🎯 BÖLÜM 2: BOOLEAN MASKING - KOŞULLU İNDEXLEME".center(70, "━"))
    
    data = torch.tensor([1, -2, 3, -4, 5, -6, 7, -8], dtype=torch.float32)
    print(f"📊 Veri: {data}\n")
    
    # Boolean mask oluşturma
    mask = data > 0
    print(f"🎭 Mask (data > 0): {mask}")
    print(f"Mask dtype: {mask.dtype}")
    print(f"Mask shape: {mask.shape}\n")
    
    # Masking ile filtreleme
    positive_values = data[mask]
    print(f"✅ Pozitif değerler: {positive_values}")
    print(f"Shape: {positive_values.shape}")
    
    # ⚠️ KRİTİK: Boolean indexing COPY oluşturur!
    print(f"\n🔴 UYARI: Boolean indexing COPY oluşturur!")
    print(f"Orijinal data pointer: {data.data_ptr()}")
    print(f"Filtered data pointer: {positive_values.data_ptr()}")
    print(f"Aynı mı? {data.data_ptr() == positive_values.data_ptr()} (HAYIR!)\n")
    
    # torch.where kullanımı
    print("─"*70)
    print("🔹 torch.where() - KOŞULLU DEĞER ATAMA")
    
    # Negatif değerleri 0 yap
    result = torch.where(data > 0, data, torch.tensor(0.0))
    print(f"torch.where(data > 0, data, 0):")
    print(f"Orijinal: {data}")
    print(f"Sonuç:    {result}")
    print(f"\n💡 where(condition, x, y) → condition True ise x, False ise y")
    
    # Çok boyutlu masking
    print("\n" + "─"*70)
    print("🔹 ÇOK BOYUTLU MASKING")
    
    matrix = torch.randn(4, 5)
    print(f"Matris:\n{matrix}")
    
    # 0'dan büyük elemanları bul
    mask_2d = matrix > 0
    print(f"\nMask (matrix > 0):\n{mask_2d}")
    
    positive_elements = matrix[mask_2d]
    print(f"\nPozitif elemanlar (1D): {positive_elements}")
    print(f"Shape: {positive_elements.shape} (Düzleştirildi!)")
    
    # In-place masking
    print("\n" + "─"*70)
    print("🔹 IN-PLACE MASKING")
    
    matrix_copy = matrix.clone()
    matrix_copy[matrix_copy < 0] = 0  # Negatif değerleri sıfırla
    print(f"Negatifler sıfırlandı:\n{matrix_copy}")


def demonstrate_fancy_indexing() -> None:
    """
    Fancy indexing (tensor indexing) tekniklerini gösterir.
    """
    print("\n" + "🎯 BÖLÜM 3: FANCY INDEXING - TENSOR İLE İNDEXLEME".center(70, "━"))
    
    data = torch.arange(10, 20)  # [10, 11, 12, ..., 19]
    print(f"📊 Veri: {data}\n")
    
    # Integer tensor ile indexleme
    indices = torch.tensor([0, 2, 5, 7])
    selected = data[indices]
    print(f"🔹 INTEGER TENSOR INDEXING")
    print(f"Indices: {indices}")
    print(f"data[indices]: {selected}")
    print(f"\n⚠️  Bu da COPY oluşturur!")
    print(f"Aynı storage? {data.data_ptr() == selected.data_ptr()} (HAYIR!)\n")
    
    # 2D fancy indexing
    print("─"*70)
    print("🔹 2D FANCY INDEXING")
    
    matrix = torch.arange(20).reshape(4, 5)
    print(f"Matris (4×5):\n{matrix}\n")
    
    # Belirli satırları seç
    row_indices = torch.tensor([0, 2, 3])
    selected_rows = matrix[row_indices]
    print(f"Satır indices: {row_indices}")
    print(f"Seçilen satırlar:\n{selected_rows}\n")
    
    # Satır VE sütun indexleme
    row_idx = torch.tensor([0, 1, 2, 3])
    col_idx = torch.tensor([0, 2, 4, 1])
    
    diagonal_elements = matrix[row_idx, col_idx]
    print(f"Satır indices: {row_idx}")
    print(f"Sütun indices: {col_idx}")
    print(f"Seçilen elemanlar: {diagonal_elements}")
    print(f"💡 matrix[i, j] → [matrix[0,0], matrix[1,2], matrix[2,4], matrix[3,1]]")
    
    # Advanced: Broadcasting ile fancy indexing
    print("\n" + "─"*70)
    print("🔹 BROADCASTING + FANCY INDEXING")
    
    # Her satırdan farklı sütunları seç
    row_idx = torch.arange(4).unsqueeze(1)  # (4, 1)
    col_idx = torch.tensor([[0, 2], [1, 3], [2, 4], [0, 1]])  # (4, 2)
    
    result = matrix[row_idx, col_idx]
    print(f"Row indices (4×1):\n{row_idx}")
    print(f"Col indices (4×2):\n{col_idx}")
    print(f"Sonuç (4×2):\n{result}")


def demonstrate_view_vs_copy() -> None:
    """
    View ve Copy arasındaki kritik farkları gösterir.
    """
    print("\n" + "🎯 BÖLÜM 4: VIEW VS COPY - BELLEK PAYLAŞIMI".center(70, "━"))
    
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print(f"📊 Orijinal Tensor:\n{original}\n")
    
    # VIEW oluşturan işlemler
    print("─"*70)
    print("✅ VIEW OLUŞTURAN İŞLEMLER (Storage paylaşımı)")
    
    operations = [
        ("Slice", original[1:3]),
        ("Transpose", original.t()),
        ("View", original.view(4, 3)),
        ("Reshape (contiguous)", original.reshape(2, 6)),
        ("Narrow", original.narrow(0, 0, 2)),
        ("Expand", original[:, :2].expand(3, 4)),
    ]
    
    for name, tensor in operations:
        is_same_storage = original.data_ptr() == tensor.data_ptr()
        print(f"{name:20} → Storage paylaşımı: {is_same_storage}")
    
    # COPY oluşturan işlemler
    print("\n" + "─"*70)
    print("❌ COPY OLUŞTURAN İŞLEMLER (Yeni storage)")
    
    copy_operations = [
        ("Clone", original.clone()),
        ("Boolean Indexing", original[original > 5]),
        ("Fancy Indexing", original[torch.tensor([0, 2])]),
        ("Contiguous", original.t().contiguous()),
        ("Detach + Clone", original.detach().clone()),
    ]
    
    for name, tensor in copy_operations:
        is_same_storage = original.data_ptr() == tensor.data_ptr()
        print(f"{name:20} → Storage paylaşımı: {is_same_storage}")
    
    # View'da değişiklik yapma
    print("\n" + "─"*70)
    print("🔴 VIEW'DA DEĞİŞİKLİK YAPMA TESTİ")
    
    view_tensor = original[0, :]  # İlk satır (view)
    print(f"View (ilk satır): {view_tensor}")
    
    view_tensor[0] = 999
    print(f"View değiştirildi → view_tensor[0] = 999")
    print(f"Orijinal tensor:\n{original}")
    print(f"💡 Orijinal de değişti! (Aynı storage)")
    
    # Copy'de değişiklik yapma
    print("\n" + "─"*70)
    print("✅ COPY'DE DEĞİŞİKLİK YAPMA TESTİ")
    
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4)  # Reset
    copy_tensor = original.clone()
    
    copy_tensor[0, 0] = 777
    print(f"Copy değiştirildi → copy_tensor[0,0] = 777")
    print(f"Orijinal tensor:\n{original}")
    print(f"💡 Orijinal değişmedi! (Farklı storage)")


def demonstrate_advanced_techniques() -> None:
    """
    Gelişmiş indexleme teknikleri ve optimizasyonlar.
    """
    print("\n" + "🎯 BÖLÜM 5: GELİŞMİŞ TEKNİKLER VE OPTİMİZASYONLAR".center(70, "━"))
    
    # torch.masked_select
    print("🔹 torch.masked_select() - MASKING İLE SEÇME")
    
    data = torch.randn(3, 4)
    mask = data > 0
    
    selected = torch.masked_select(data, mask)
    print(f"Data:\n{data}")
    print(f"Mask:\n{mask}")
    print(f"Seçilen elemanlar: {selected}")
    print(f"Shape: {selected.shape} (1D!)\n")
    
    # torch.masked_fill
    print("─"*70)
    print("🔹 torch.masked_fill() - MASKING İLE DOLDURMA")
    
    data_copy = data.clone()
    data_copy.masked_fill_(mask, 0.0)
    print(f"Pozitif değerler 0 yapıldı:\n{data_copy}\n")
    
    # torch.index_select
    print("─"*70)
    print("🔹 torch.index_select() - BOYUT BAZLI SEÇME")
    
    matrix = torch.arange(20).reshape(4, 5)
    indices = torch.tensor([0, 2, 3])
    
    selected = torch.index_select(matrix, dim=0, index=indices)
    print(f"Matris:\n{matrix}")
    print(f"Dim 0'da indices {indices} seçildi:\n{selected}\n")
    
    # torch.gather
    print("─"*70)
    print("🔹 torch.gather() - GELİŞMİŞ TOPLAMA")
    
    scores = torch.tensor([
        [0.1, 0.3, 0.6],
        [0.4, 0.2, 0.4],
        [0.7, 0.1, 0.2]
    ])
    
    # Her satırdan en yüksek skorun indexini bul
    max_indices = scores.argmax(dim=1, keepdim=True)
    print(f"Scores:\n{scores}")
    print(f"Max indices (dim=1):\n{max_indices}")
    
    # gather ile en yüksek skorları al
    max_scores = torch.gather(scores, dim=1, index=max_indices)
    print(f"Max scores:\n{max_scores}\n")
    
    # Performans karşılaştırması
    print("─"*70)
    print("🔹 PERFORMANS KARŞILAŞTIRMASI")
    
    big_tensor = torch.randn(10000, 1000)
    mask = big_tensor > 0
    
    # Yöntem 1: Boolean indexing
    start = time.time()
    result1 = big_tensor[mask]
    time1 = time.time() - start
    
    # Yöntem 2: masked_select
    start = time.time()
    result2 = torch.masked_select(big_tensor, mask)
    time2 = time.time() - start
    
    # Yöntem 3: where + flatten
    start = time.time()
    result3 = torch.where(mask, big_tensor, torch.tensor(float('nan'))).flatten()
    result3 = result3[~torch.isnan(result3)]
    time3 = time.time() - start
    
    print(f"Boolean indexing:  {time1:.6f}s")
    print(f"masked_select:     {time2:.6f}s")
    print(f"where + flatten:   {time3:.6f}s")
    print(f"\n💡 En hızlı: {'Boolean indexing' if time1 < min(time2, time3) else 'masked_select' if time2 < time3 else 'where + flatten'}")


def demonstrate_common_pitfalls() -> None:
    """
    Sık yapılan hataları gösterir.
    """
    print("\n" + "🎯 BÖLÜM 6: YAYGIN HATALAR VE ÇÖZÜMLER".center(70, "━"))
    
    # HATA 1: View üzerinde in-place işlem
    print("🔴 HATA 1: View Üzerinde In-place İşlem")
    
    original = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    original.requires_grad = True
    
    view = original[0, :]
    
    try:
        # YANLIŞ: View üzerinde in-place işlem gradient graph'ı bozar
        # view.add_(1.0)  # Bu satır açılırsa backward() hatası verir
        print("⚠️  view.add_(1.0) gradient graph'ı bozar!")
        
        # DOĞRU: Yeni tensor döndür
        new_view = view.add(1.0)
        print(f"✅ Doğru: new_view = view.add(1.0)")
    except RuntimeError as e:
        print(f"❌ HATA: {e}")
    
    # HATA 2: Boolean indexing ile assignment
    print("\n" + "─"*70)
    print("🔴 HATA 2: Boolean Indexing ile Assignment")
    
    data = torch.randn(5)
    mask = data > 0
    
    # YANLIŞ: Boolean indexing copy oluşturur
    # data[mask] = 0  # Bu çalışır ama dikkatli ol!
    
    # DOĞRU: masked_fill_ kullan
    data.masked_fill_(mask, 0.0)
    print(f"✅ Doğru: data.masked_fill_(mask, 0.0)")
    
    # HATA 3: Fancy indexing ile gradient
    print("\n" + "─"*70)
    print("🔴 HATA 3: Fancy Indexing ile Gradient")
    
    embeddings = torch.randn(100, 50, requires_grad=True)
    indices = torch.tensor([0, 5, 10])
    
    selected = embeddings[indices]
    loss = selected.sum()
    loss.backward()
    
    print(f"Embeddings gradient shape: {embeddings.grad.shape}")
    print(f"Non-zero gradients: {(embeddings.grad != 0).sum().item()}")
    print(f"💡 Sadece seçilen satırlarda gradient var (sparse gradient)")


def main() -> None:
    """
    Ana çalıştırma fonksiyonu.
    """
    print("\n" + "="*70)
    print("🚀 GELİŞMİŞ İNDEXLEME - MASKING VE FANCY INDEXING".center(70))
    print("="*70)
    
    demonstrate_basic_indexing()
    demonstrate_boolean_masking()
    demonstrate_fancy_indexing()
    demonstrate_view_vs_copy()
    demonstrate_advanced_techniques()
    demonstrate_common_pitfalls()
    
    print("\n" + "="*70)
    print("✅ DERS 03 TAMAMLANDI!".center(70))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
