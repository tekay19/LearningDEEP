"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DERS 01: TENSOR MEKANİĞİ - BELLEK DÜZENİ VE STRIDE ANALİZİ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Amaç: PyTorch Tensor'larının NumPy dizilerinden farkını anlamak.
Storage, Offset ve Stride kavramlarını bellek düzeyinde incelemek.

Hedef Kitle: Senior Developer'lar için "Under the Hood" analiz.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import numpy as np
from typing import Tuple, Any
import sys


def inspect_tensor_anatomy(tensor: torch.Tensor, name: str = "Tensor") -> None:
    """
    Bir tensörün tüm anatomik özelliklerini detaylı şekilde yazdırır.
    
    Args:
        tensor: İncelenecek PyTorch tensörü
        name: Tensörün tanımlayıcı ismi
    """
    print(f"\n{'='*70}")
    print(f"🔬 {name} ANATOMİK ANALİZ")
    print(f"{'='*70}")
    print(f"📊 Shape (Boyut):        {tensor.shape}")
    print(f"🧮 Dtype (Veri Tipi):    {tensor.dtype}")
    print(f"📏 Stride (Adım):        {tensor.stride()}")
    print(f"💾 Storage Size:         {tensor.storage().size()} elements")
    print(f"📍 Storage Offset:       {tensor.storage_offset()}")
    print(f"🖥️  Device (Cihaz):       {tensor.device}")
    print(f"🎓 Requires Grad:        {tensor.requires_grad}")
    print(f"🔗 Is Contiguous:        {tensor.is_contiguous()}")
    print(f"💽 Memory (bytes):       {tensor.element_size() * tensor.nelement()}")
    print(f"{'='*70}\n")


def demonstrate_tensor_vs_numpy() -> None:
    """
    PyTorch Tensor ile NumPy Array arasındaki temel farkları gösterir.
    Özellikle GPU desteği ve autograd özelliklerini vurgular.
    """
    print("\n" + "🎯 BÖLÜM 1: TENSOR VS NUMPY - TEMEL FARKLAR".center(70, "━"))
    
    # NumPy array oluşturma
    np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    print(f"\n📦 NumPy Array:\n{np_array}")
    print(f"Type: {type(np_array)}, Dtype: {np_array.dtype}")
    
    # PyTorch tensor oluşturma (NumPy'dan)
    tensor_from_numpy = torch.from_numpy(np_array)
    inspect_tensor_anatomy(tensor_from_numpy, "NumPy'dan Dönüştürülmüş Tensor")
    
    # ⚠️ KRİTİK: NumPy ve Tensor aynı belleği paylaşır!
    print("🔴 BELLEK PAYLAŞIMI TESTİ:")
    np_array[0, 0] = 999
    print(f"NumPy değiştirildi -> np_array[0,0] = {np_array[0, 0]}")
    print(f"Tensor otomatik güncellendi -> tensor[0,0] = {tensor_from_numpy[0, 0]}")
    print("⚡ Sonuç: Aynı bellek bölgesini gösteriyorlar (Zero-copy operation)\n")
    
    # Sıfırdan PyTorch tensor oluşturma
    pure_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], 
                                dtype=torch.float32, 
                                requires_grad=True)  # Gradient takibi aktif
    inspect_tensor_anatomy(pure_tensor, "Saf PyTorch Tensor (Gradient Aktif)")


def demonstrate_storage_and_offset() -> None:
    """
    PyTorch'un Storage mekanizmasını ve Offset kavramını açıklar.
    Birden fazla tensor'un aynı storage'ı nasıl paylaştığını gösterir.
    """
    print("\n" + "🎯 BÖLÜM 2: STORAGE VE OFFSET - BELLEK OPTİMİZASYONU".center(70, "━"))
    
    # Ana tensor oluştur
    original = torch.arange(12, dtype=torch.float32)  # [0, 1, 2, ..., 11]
    inspect_tensor_anatomy(original, "Orijinal Tensor")
    
    # Storage içeriğini göster
    print("💾 STORAGE İÇERİĞİ (Ham Bellek):")
    print(f"Storage Data Pointer: {original.data_ptr()}")
    print(f"Storage içeriği: {list(original.storage())}\n")
    
    # View ile yeniden şekillendirme (AYNI STORAGE)
    reshaped = original.view(3, 4)  # 3x4 matris
    inspect_tensor_anatomy(reshaped, "View ile Yeniden Şekillendirilmiş (3x4)")
    
    # ⚠️ KRİTİK: Her iki tensor de aynı storage'ı kullanıyor
    print("🔴 STORAGE PAYLAŞIMI TESTİ:")
    print(f"Orijinal Storage ID: {original.storage().data_ptr()}")
    print(f"Reshaped Storage ID: {reshaped.storage().data_ptr()}")
    print(f"Aynı mı? {original.storage().data_ptr() == reshaped.storage().data_ptr()}")
    
    # Slicing ile offset değişimi
    sliced = original[3:9]  # Index 3'ten 9'a kadar
    inspect_tensor_anatomy(sliced, "Slice Edilmiş Tensor [3:9]")
    
    print("📍 OFFSET FARKI:")
    print(f"Orijinal offset: {original.storage_offset()}")
    print(f"Sliced offset: {sliced.storage_offset()}")
    print(f"⚡ Slice, storage'da 3. elemandan başlıyor (zero-copy!)\n")


def demonstrate_stride_mechanism() -> None:
    """
    Stride (adım) mekanizmasını detaylı açıklar.
    Transpose ve permute işlemlerinin stride'ı nasıl değiştirdiğini gösterir.
    """
    print("\n" + "🎯 BÖLÜM 3: STRIDE MEKANİZMASI - BELLEK ATLAMALARI".center(70, "━"))
    
    # 2D tensor oluştur
    matrix = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print(f"📊 Orijinal Matris (3x4):\n{matrix}")
    inspect_tensor_anatomy(matrix, "Orijinal Matris")
    
    print("🧮 STRIDE HESAPLAMA:")
    print(f"Stride: {matrix.stride()}")
    print(f"  - Satır değiştirmek için 4 eleman atla (stride[0]=4)")
    print(f"  - Sütun değiştirmek için 1 eleman atla (stride[1]=1)")
    print(f"  - matrix[1,2] konumu = base + 1*4 + 2*1 = 0 + 4 + 2 = 6. eleman")
    print(f"  - Doğrulama: matrix[1,2] = {matrix[1, 2]} (beklenen: 6.0)\n")
    
    # Transpose işlemi
    transposed = matrix.t()  # veya matrix.transpose(0, 1)
    print(f"📊 Transpose Edilmiş Matris (4x3):\n{transposed}")
    inspect_tensor_anatomy(transposed, "Transpose Edilmiş Matris")
    
    print("⚠️ KRİTİK NOKTA:")
    print(f"Transpose sonrası stride: {transposed.stride()}")
    print(f"  - Stride ters döndü: (1, 4) -> Artık ROW-MAJOR değil!")
    print(f"  - Contiguous mu? {transposed.is_contiguous()}")
    print(f"  - Bellekte veri AYNI, sadece erişim şekli değişti!\n")
    
    # HATA ÖRNEĞİ: Non-contiguous tensor'da view kullanımı
    print("🔴 YAYGIN HATA ÖRNEĞİ:")
    try:
        # HATA: Transpose edilmiş tensor contiguous değil, view() çalışmaz
        wrong_view = transposed.view(12)
        print(f"View başarılı: {wrong_view}")
    except RuntimeError as e:
        print(f"❌ HATA: {e}")
        print(f"💡 ÇÖZÜM: Önce .contiguous() çağır!")
        correct_view = transposed.contiguous().view(12)
        print(f"✅ Doğru kullanım: {correct_view}\n")


def demonstrate_contiguous_memory() -> None:
    """
    Contiguous (bitişik) bellek kavramını açıklar.
    .contiguous() metodunun ne zaman gerekli olduğunu gösterir.
    """
    print("\n" + "🎯 BÖLÜM 4: CONTIGUOUS MEMORY - BİTİŞİK BELLEK".center(70, "━"))
    
    # Contiguous tensor
    cont_tensor = torch.arange(6).reshape(2, 3)
    print(f"📊 Contiguous Tensor:\n{cont_tensor}")
    print(f"Is contiguous? {cont_tensor.is_contiguous()}")
    print(f"Stride: {cont_tensor.stride()}")
    print(f"Bellekte sıralama: [0,1,2,3,4,5] (Row-major order)\n")
    
    # Non-contiguous tensor (transpose sonrası)
    non_cont = cont_tensor.t()
    print(f"📊 Non-Contiguous Tensor (Transpose):\n{non_cont}")
    print(f"Is contiguous? {non_cont.is_contiguous()}")
    print(f"Stride: {non_cont.stride()}")
    print(f"Bellekte sıralama: Hala [0,1,2,3,4,5] ama erişim farklı!\n")
    
    # Contiguous hale getirme
    made_contiguous = non_cont.contiguous()
    print(f"📊 Contiguous Yapılmış Tensor:\n{made_contiguous}")
    print(f"Is contiguous? {made_contiguous.is_contiguous()}")
    print(f"Stride: {made_contiguous.stride()}")
    
    print("⚡ PERFORMANS ETKİSİ:")
    print(f"Non-contiguous data pointer: {non_cont.data_ptr()}")
    print(f"Contiguous data pointer: {made_contiguous.data_ptr()}")
    print(f"Farklı mı? {non_cont.data_ptr() != made_contiguous.data_ptr()}")
    print(f"💡 .contiguous() YENİ BELLEK AYIRIR ve veriyi kopyalar!\n")


def demonstrate_memory_efficiency() -> None:
    """
    View vs Clone vs Copy işlemlerinin bellek kullanımını karşılaştırır.
    """
    print("\n" + "🎯 BÖLÜM 5: VIEW VS CLONE VS COPY - BELLEK VERİMLİLİĞİ".center(70, "━"))
    
    original = torch.arange(1000000, dtype=torch.float32)  # 1 milyon eleman
    original_size = original.element_size() * original.nelement()
    
    print(f"📊 Orijinal Tensor: {original.shape}")
    print(f"💾 Bellek kullanımı: {original_size / (1024**2):.2f} MB\n")
    
    # VIEW: Aynı belleği paylaşır
    viewed = original.view(1000, 1000)
    print(f"🔗 VIEW İşlemi:")
    print(f"  - Yeni shape: {viewed.shape}")
    print(f"  - Aynı storage? {original.data_ptr() == viewed.data_ptr()}")
    print(f"  - Ekstra bellek: 0 MB (Zero-copy!)\n")
    
    # CLONE: Yeni bellek ayırır, gradient graph korunur
    cloned = original.clone()
    print(f"📋 CLONE İşlemi:")
    print(f"  - Aynı storage? {original.data_ptr() == cloned.data_ptr()}")
    print(f"  - Ekstra bellek: {(cloned.element_size() * cloned.nelement()) / (1024**2):.2f} MB")
    print(f"  - Gradient graph korunur mu? Evet (autograd için kullan)\n")
    
    # DETACH + CLONE: Gradient graph kopmaz
    detached = original.detach().clone()
    print(f"✂️ DETACH + CLONE:")
    print(f"  - Gradient graph'tan koptu mu? Evet")
    print(f"  - Kullanım: Inference sırasında bellek tasarrufu\n")


def intentional_bug_demo() -> None:
    """
    Yeni başlayanların sık yaptığı hataları gösterir ve düzeltir.
    """
    print("\n" + "🎯 BONUS: YAYGIN HATALAR VE ÇÖZÜMLER".center(70, "━"))
    
    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    b = torch.tensor([4, 5, 6], dtype=torch.float32)
    
    # HATA 1: Element-wise çarpma vs Dot product
    print("🔴 HATA 1: Çarpma İşlemi Karışıklığı")
    element_wise = a * b  # Element-wise multiplication
    print(f"a * b (Element-wise): {element_wise}")
    
    # DOĞRU: Dot product için @ veya torch.dot
    dot_product = a @ b  # veya torch.dot(a, b)
    print(f"a @ b (Dot product): {dot_product}")
    print(f"💡 Fark: * -> [1*4, 2*5, 3*6], @ -> 1*4 + 2*5 + 3*6\n")
    
    # HATA 2: In-place işlem sonrası gradient hatası
    print("🔴 HATA 2: In-place İşlem Gradient Hatası")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    print(f"Orijinal x: {x}")
    
    # YANLIŞ: In-place işlem gradient graph'ı bozar
    # x.add_(1.0)  # Bu satır açılırsa backward() hatası verir
    
    # DOĞRU: Yeni tensor döndür
    x_new = x.add(1.0)  # veya x = x + 1.0
    print(f"x + 1.0 (Doğru): {x_new}")
    print(f"💡 In-place işlemler (_ile bitenler) gradient'i bozar!\n")


def main() -> None:
    """
    Ana çalıştırma fonksiyonu - Tüm demoları sırayla çalıştırır.
    """
    print("\n" + "="*70)
    print("🚀 PYTORCH TENSOR MEKANİĞİ - BELLEK DÜZENİ ANALİZİ".center(70))
    print("="*70)
    
    demonstrate_tensor_vs_numpy()
    demonstrate_storage_and_offset()
    demonstrate_stride_mechanism()
    demonstrate_contiguous_memory()
    demonstrate_memory_efficiency()
    intentional_bug_demo()
    
    print("\n" + "="*70)
    print("✅ DERS 01 TAMAMLANDI!".center(70))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
