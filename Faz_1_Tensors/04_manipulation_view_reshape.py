"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DERS 04: TENSOR MANİPÜLASYONU - VIEW, RESHAPE, PERMUTE, TRANSPOSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Amaç: Tensor şekil değiştirme işlemlerini derinlemesine anlamak.
Contiguous bellek sorunsalını çözmek.

Hedef Kitle: Senior Developer'lar için "Under the Hood" analiz.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import numpy as np
from typing import Tuple, List
import time


def demonstrate_view_vs_reshape() -> None:
    """
    view() ve reshape() arasındaki kritik farkları gösterir.
    """
    print("\n" + "🎯 BÖLÜM 1: VIEW VS RESHAPE - NE ZAMAN HANGİSİ?".center(70, "━"))
    
    # Contiguous tensor
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print(f"📊 Orijinal Tensor (3×4):\n{tensor}")
    print(f"Is contiguous: {tensor.is_contiguous()}")
    print(f"Stride: {tensor.stride()}\n")
    
    # VIEW: Contiguous tensor'da çalışır
    print("─"*70)
    print("✅ VIEW - Contiguous Tensor'da Çalışır")
    
    viewed = tensor.view(4, 3)
    print(f"tensor.view(4, 3):\n{viewed}")
    print(f"Storage paylaşımı: {tensor.data_ptr() == viewed.data_ptr()}")
    print(f"Stride: {viewed.stride()}\n")
    
    # Non-contiguous tensor
    print("─"*70)
    print("🔴 VIEW - Non-Contiguous Tensor'da HATA")
    
    transposed = tensor.t()
    print(f"Transposed tensor:\n{transposed}")
    print(f"Is contiguous: {transposed.is_contiguous()}")
    print(f"Stride: {transposed.stride()}")
    
    try:
        # HATA: Non-contiguous tensor'da view() çalışmaz
        wrong_view = transposed.view(12)
    except RuntimeError as e:
        print(f"\n❌ HATA: {e}")
        print(f"\n💡 ÇÖZÜM 1: .contiguous() kullan")
        correct_view = transposed.contiguous().view(12)
        print(f"transposed.contiguous().view(12): {correct_view}")
        print(f"Yeni storage oluşturuldu: {tensor.data_ptr() != correct_view.data_ptr()}")
    
    # RESHAPE: Her zaman çalışır
    print("\n" + "─"*70)
    print("✅ RESHAPE - Her Zaman Çalışır")
    
    reshaped = transposed.reshape(12)
    print(f"transposed.reshape(12): {reshaped}")
    print(f"Storage paylaşımı: {transposed.data_ptr() == reshaped.data_ptr()}")
    print(f"\n💡 reshape() gerekirse otomatik .contiguous() çağırır")
    
    # Performans karşılaştırması
    print("\n" + "─"*70)
    print("⏱️  PERFORMANS KARŞILAŞTIRMASI")
    
    big_tensor = torch.randn(1000, 1000)
    
    # view() - Zero-copy
    start = time.time()
    for _ in range(10000):
        _ = big_tensor.view(1000000)
    view_time = time.time() - start
    
    # reshape() - Contiguous tensor'da zero-copy
    start = time.time()
    for _ in range(10000):
        _ = big_tensor.reshape(1000000)
    reshape_time = time.time() - start
    
    print(f"view() 10000 kez:    {view_time:.6f}s")
    print(f"reshape() 10000 kez: {reshape_time:.6f}s")
    print(f"Fark: ~{abs(view_time - reshape_time):.6f}s (İhmal edilebilir)")


def demonstrate_permute_and_transpose() -> None:
    """
    permute() ve transpose() işlemlerini detaylı açıklar.
    """
    print("\n" + "🎯 BÖLÜM 2: PERMUTE VE TRANSPOSE - BOYUT YER DEĞİŞTİRME".center(70, "━"))
    
    # 3D tensor
    tensor = torch.arange(24).reshape(2, 3, 4)
    print(f"📊 Orijinal Tensor (2×3×4):\n{tensor}")
    print(f"Shape: {tensor.shape}")
    print(f"Stride: {tensor.stride()}\n")
    
    # TRANSPOSE: İki boyutu değiştir
    print("─"*70)
    print("🔹 TRANSPOSE - İki Boyut Değiştir")
    
    transposed = tensor.transpose(0, 2)  # Dim 0 ve 2'yi değiştir
    print(f"tensor.transpose(0, 2):")
    print(f"Shape: {tensor.shape} → {transposed.shape}")
    print(f"Stride: {tensor.stride()} → {transposed.stride()}")
    print(f"Is contiguous: {transposed.is_contiguous()}")
    print(f"Storage paylaşımı: {tensor.data_ptr() == transposed.data_ptr()}\n")
    
    # PERMUTE: Tüm boyutları yeniden sırala
    print("─"*70)
    print("🔹 PERMUTE - Tüm Boyutları Yeniden Sırala")
    
    permuted = tensor.permute(2, 0, 1)  # (2,3,4) → (4,2,3)
    print(f"tensor.permute(2, 0, 1):")
    print(f"Shape: {tensor.shape} → {permuted.shape}")
    print(f"Stride: {tensor.stride()} → {permuted.stride()}")
    print(f"Is contiguous: {permuted.is_contiguous()}")
    
    # Stride hesaplama doğrulaması
    print(f"\n🧮 STRIDE HESAPLAMA:")
    print(f"Orijinal stride: {tensor.stride()} → (12, 4, 1)")
    print(f"  - Dim 0: 3×4 = 12 eleman atla")
    print(f"  - Dim 1: 4 eleman atla")
    print(f"  - Dim 2: 1 eleman atla")
    print(f"\nPermute sonrası: {permuted.stride()} → (1, 12, 4)")
    print(f"  - Yeni dim 0 (eski dim 2): stride = 1")
    print(f"  - Yeni dim 1 (eski dim 0): stride = 12")
    print(f"  - Yeni dim 2 (eski dim 1): stride = 4")
    
    # Pratik örnek: Image tensor (NCHW → NHWC)
    print("\n" + "─"*70)
    print("🔹 PRATİK ÖRNEK: Image Tensor Dönüşümü")
    
    # PyTorch format: (Batch, Channels, Height, Width)
    image_nchw = torch.randn(32, 3, 224, 224)
    print(f"PyTorch format (NCHW): {image_nchw.shape}")
    
    # TensorFlow format: (Batch, Height, Width, Channels)
    image_nhwc = image_nchw.permute(0, 2, 3, 1)
    print(f"TensorFlow format (NHWC): {image_nhwc.shape}")
    print(f"Is contiguous: {image_nhwc.is_contiguous()}")
    print(f"\n💡 ONNX export için .contiguous() gerekebilir!")


def demonstrate_squeeze_and_unsqueeze() -> None:
    """
    squeeze() ve unsqueeze() ile boyut ekleme/çıkarma.
    """
    print("\n" + "🎯 BÖLÜM 3: SQUEEZE VE UNSQUEEZE - BOYUT EKLEME/ÇIKARMA".center(70, "━"))
    
    # UNSQUEEZE: Boyut ekle
    print("🔹 UNSQUEEZE - Boyut Ekle")
    
    tensor = torch.tensor([1, 2, 3, 4])
    print(f"Orijinal: {tensor.shape}")
    
    unsqueezed_0 = tensor.unsqueeze(0)
    print(f"unsqueeze(0): {unsqueezed_0.shape} → {unsqueezed_0}")
    
    unsqueezed_1 = tensor.unsqueeze(1)
    print(f"unsqueeze(1): {unsqueezed_1.shape} →\n{unsqueezed_1}")
    
    unsqueezed_neg = tensor.unsqueeze(-1)
    print(f"unsqueeze(-1): {unsqueezed_neg.shape} (Son boyuta ekle)\n")
    
    # SQUEEZE: 1 boyutundaki dimensionları kaldır
    print("─"*70)
    print("🔹 SQUEEZE - 1 Boyutundaki Dimensionları Kaldır")
    
    tensor_with_ones = torch.randn(1, 3, 1, 5, 1)
    print(f"Orijinal: {tensor_with_ones.shape}")
    
    squeezed_all = tensor_with_ones.squeeze()
    print(f"squeeze() (tümü): {squeezed_all.shape}")
    
    squeezed_dim = tensor_with_ones.squeeze(0)
    print(f"squeeze(0): {squeezed_dim.shape}")
    
    squeezed_dim2 = tensor_with_ones.squeeze(2)
    print(f"squeeze(2): {squeezed_dim2.shape}\n")
    
    # Pratik kullanım: Batch dimension ekleme
    print("─"*70)
    print("🔹 PRATİK KULLANIM: Batch Dimension")
    
    single_image = torch.randn(3, 224, 224)  # (C, H, W)
    print(f"Tek görüntü: {single_image.shape}")
    
    batched = single_image.unsqueeze(0)  # (1, C, H, W)
    print(f"Batch'e eklendi: {batched.shape}")
    print(f"💡 Model'e tek görüntü göndermek için gerekli!")


def demonstrate_flatten_and_unflatten() -> None:
    """
    flatten() ve unflatten() ile tensor düzleştirme.
    """
    print("\n" + "🎯 BÖLÜM 4: FLATTEN VE UNFLATTEN - DÜZLEŞTIRME".center(70, "━"))
    
    # FLATTEN
    print("🔹 FLATTEN - Tensor Düzleştirme")
    
    tensor = torch.arange(24).reshape(2, 3, 4)
    print(f"Orijinal (2×3×4):\n{tensor}\n")
    
    # Tüm boyutları düzleştir
    flat_all = tensor.flatten()
    print(f"flatten(): {flat_all.shape}")
    print(f"Sonuç: {flat_all}\n")
    
    # Belirli boyutları düzleştir
    flat_partial = tensor.flatten(start_dim=1)
    print(f"flatten(start_dim=1): {flat_partial.shape}")
    print(f"Sonuç:\n{flat_partial}")
    print(f"💡 İlk boyut korundu, geri kalanlar düzleştirildi\n")
    
    # UNFLATTEN (PyTorch 1.13+)
    print("─"*70)
    print("🔹 UNFLATTEN - Düzleştirilmiş Tensor'u Geri Al")
    
    flat = torch.arange(24)
    print(f"Düzleştirilmiş: {flat.shape}")
    
    unflat = flat.unflatten(0, (2, 3, 4))
    print(f"unflatten(0, (2,3,4)): {unflat.shape}")
    print(f"Sonuç:\n{unflat}\n")
    
    # CNN'de kullanım
    print("─"*70)
    print("🔹 PRATİK: CNN → Fully Connected Geçişi")
    
    # Conv layer çıktısı: (Batch, Channels, H, W)
    conv_output = torch.randn(32, 512, 7, 7)
    print(f"Conv output: {conv_output.shape}")
    
    # Fully connected için düzleştir
    fc_input = conv_output.flatten(start_dim=1)
    print(f"FC input: {fc_input.shape}")
    print(f"💡 Batch dimension korundu, geri kalanlar düzleştirildi")


def demonstrate_advanced_manipulations() -> None:
    """
    Gelişmiş manipülasyon teknikleri.
    """
    print("\n" + "🎯 BÖLÜM 5: GELİŞMİŞ MANİPÜLASYONLAR".center(70, "━"))
    
    # CHUNK: Tensor'u parçalara böl
    print("🔹 CHUNK - Tensor'u Eşit Parçalara Böl")
    
    tensor = torch.arange(12).reshape(3, 4)
    print(f"Orijinal:\n{tensor}\n")
    
    chunks = tensor.chunk(2, dim=0)  # 2 parçaya böl (dim=0)
    print(f"chunk(2, dim=0): {len(chunks)} parça")
    for i, chunk in enumerate(chunks):
        print(f"Parça {i}: {chunk.shape}\n{chunk}\n")
    
    # SPLIT: Tensor'u belirli boyutlarda böl
    print("─"*70)
    print("🔹 SPLIT - Tensor'u Belirli Boyutlarda Böl")
    
    splits = tensor.split([1, 2], dim=0)  # 1 ve 2 satırlık parçalar
    print(f"split([1, 2], dim=0): {len(splits)} parça")
    for i, split in enumerate(splits):
        print(f"Parça {i}: {split.shape}\n{split}\n")
    
    # CAT: Tensor'ları birleştir
    print("─"*70)
    print("🔹 CAT - Tensor'ları Birleştir")
    
    t1 = torch.tensor([[1, 2], [3, 4]])
    t2 = torch.tensor([[5, 6], [7, 8]])
    
    cat_dim0 = torch.cat([t1, t2], dim=0)
    print(f"cat([t1, t2], dim=0): {cat_dim0.shape}\n{cat_dim0}\n")
    
    cat_dim1 = torch.cat([t1, t2], dim=1)
    print(f"cat([t1, t2], dim=1): {cat_dim1.shape}\n{cat_dim1}\n")
    
    # STACK: Yeni boyut ekleyerek birleştir
    print("─"*70)
    print("🔹 STACK - Yeni Boyut Ekleyerek Birleştir")
    
    stacked_dim0 = torch.stack([t1, t2], dim=0)
    print(f"stack([t1, t2], dim=0): {stacked_dim0.shape}\n{stacked_dim0}\n")
    
    stacked_dim1 = torch.stack([t1, t2], dim=1)
    print(f"stack([t1, t2], dim=1): {stacked_dim1.shape}\n{stacked_dim1}")
    
    print(f"\n💡 cat vs stack:")
    print(f"  - cat: Mevcut boyutta birleştir")
    print(f"  - stack: Yeni boyut ekleyerek birleştir")


def demonstrate_common_pitfalls() -> None:
    """
    Sık yapılan hataları gösterir.
    """
    print("\n" + "🎯 BÖLÜM 6: YAYGIN HATALAR VE ÇÖZÜMLER".center(70, "━"))
    
    # HATA 1: view() ile boyut uyumsuzluğu
    print("🔴 HATA 1: view() Boyut Uyumsuzluğu")
    
    tensor = torch.arange(12)
    
    try:
        # YANLIŞ: Toplam eleman sayısı eşleşmiyor
        wrong_view = tensor.view(3, 5)  # 12 ≠ 15
    except RuntimeError as e:
        print(f"❌ HATA: {e}")
        print(f"\n💡 ÇÖZÜM: -1 kullan (otomatik hesaplama)")
        correct_view = tensor.view(3, -1)
        print(f"tensor.view(3, -1): {correct_view.shape}\n")
    
    # HATA 2: permute() sonrası view()
    print("─"*70)
    print("🔴 HATA 2: permute() Sonrası view()")
    
    tensor = torch.randn(2, 3, 4)
    permuted = tensor.permute(2, 0, 1)
    
    try:
        # YANLIŞ: permute() non-contiguous yapar
        wrong = permuted.view(-1)
    except RuntimeError as e:
        print(f"❌ HATA: {e}")
        print(f"\n💡 ÇÖZÜM: .contiguous() ekle")
        correct = permuted.contiguous().view(-1)
        print(f"permuted.contiguous().view(-1): {correct.shape}\n")
    
    # HATA 3: In-place işlem sonrası reshape
    print("─"*70)
    print("🔴 HATA 3: In-place İşlem Sonrası Reshape")
    
    tensor = torch.randn(3, 4, requires_grad=True)
    
    # YANLIŞ: In-place işlem gradient graph'ı bozar
    # tensor.add_(1.0)
    # reshaped = tensor.view(12)  # Gradient hatası!
    
    # DOĞRU: Yeni tensor döndür
    tensor_new = tensor.add(1.0)
    reshaped = tensor_new.view(12)
    print(f"✅ Doğru: tensor.add(1.0).view(12)")


def main() -> None:
    """
    Ana çalıştırma fonksiyonu.
    """
    print("\n" + "="*70)
    print("🚀 TENSOR MANİPÜLASYONU - VIEW, RESHAPE, PERMUTE".center(70))
    print("="*70)
    
    demonstrate_view_vs_reshape()
    demonstrate_permute_and_transpose()
    demonstrate_squeeze_and_unsqueeze()
    demonstrate_flatten_and_unflatten()
    demonstrate_advanced_manipulations()
    demonstrate_common_pitfalls()
    
    print("\n" + "="*70)
    print("✅ DERS 04 TAMAMLANDI!".center(70))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
