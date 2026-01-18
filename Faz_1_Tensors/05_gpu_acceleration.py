"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DERS 05: GPU HIZLANDIRMA - CUDA, CPU-GPU TRANSFER VE BOTTLENECK ANALİZİ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Amaç: CUDA çekirdeklerini anlamak, CPU-GPU veri transferini optimize etmek.
Performans darboğazlarını tespit etmek ve çözmek.

Hedef Kitle: Senior Developer'lar için "Under the Hood" analiz.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import numpy as np
import time
from typing import Tuple, List, Optional
import sys


def check_cuda_availability() -> None:
    """
    CUDA kullanılabilirliğini ve GPU bilgilerini gösterir.
    """
    print("\n" + "🎯 BÖLÜM 1: CUDA KULLANILABİLİRLİĞİ VE GPU Bİ̇LGİ̇LERİ̇".center(70, "━"))
    
    print(f"\n🔍 CUDA Kullanılabilir mi? {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA Versiyonu: {torch.version.cuda}")
        print(f"✅ cuDNN Versiyonu: {torch.backends.cudnn.version()}")
        print(f"✅ GPU Sayısı: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n📊 GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Toplam Bellek: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"   CUDA Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
            print(f"   Multi-Processor Count: {torch.cuda.get_device_properties(i).multi_processor_count}")
    else:
        print("⚠️  CUDA kullanılamıyor. CPU modunda çalışacağız.")
        print("💡 Google Colab veya GPU'lu bir makine kullanın.")


def demonstrate_device_management() -> None:
    """
    Tensor'ları farklı cihazlar arasında taşıma işlemlerini gösterir.
    """
    print("\n" + "🎯 BÖLÜM 2: DEVICE MANAGEMENT - Cİ̇HAZ YÖNETİ̇Mİ̇".center(70, "━"))
    
    # CPU'da tensor oluşturma
    cpu_tensor = torch.randn(3, 4)
    print(f"📊 CPU Tensor:")
    print(f"   Device: {cpu_tensor.device}")
    print(f"   Data pointer: {cpu_tensor.data_ptr()}")
    print(f"   Shape: {cpu_tensor.shape}\n")
    
    if torch.cuda.is_available():
        # GPU'ya taşıma - Yöntem 1: .to()
        print("─"*70)
        print("🔹 YÖNTEM 1: .to() ile GPU'ya Taşıma")
        
        start = time.time()
        gpu_tensor_1 = cpu_tensor.to('cuda')
        transfer_time_1 = time.time() - start
        
        print(f"   Device: {gpu_tensor_1.device}")
        print(f"   Data pointer: {gpu_tensor_1.data_ptr()}")
        print(f"   Transfer süresi: {transfer_time_1*1000:.4f} ms")
        print(f"   Yeni tensor mi? {cpu_tensor.data_ptr() != gpu_tensor_1.data_ptr()}\n")
        
        # GPU'ya taşıma - Yöntem 2: .cuda()
        print("─"*70)
        print("🔹 YÖNTEM 2: .cuda() ile GPU'ya Taşıma")
        
        gpu_tensor_2 = cpu_tensor.cuda()
        print(f"   Device: {gpu_tensor_2.device}")
        print(f"   .to('cuda') ile aynı mı? {torch.equal(gpu_tensor_1, gpu_tensor_2)}\n")
        
        # Belirli GPU'ya taşıma (Multi-GPU sistemlerde)
        if torch.cuda.device_count() > 1:
            print("─"*70)
            print("🔹 MULTI-GPU: Belirli GPU'ya Taşıma")
            
            gpu_0 = cpu_tensor.to('cuda:0')
            gpu_1 = cpu_tensor.to('cuda:1')
            
            print(f"   GPU 0: {gpu_0.device}")
            print(f"   GPU 1: {gpu_1.device}\n")
        
        # CPU'ya geri taşıma
        print("─"*70)
        print("🔹 GPU'dan CPU'ya Geri Taşıma")
        
        back_to_cpu = gpu_tensor_1.cpu()
        print(f"   Device: {back_to_cpu.device}")
        print(f"   Orijinal ile aynı değerler mi? {torch.equal(cpu_tensor, back_to_cpu)}\n")
        
        # ⚠️ KRİTİK: Farklı cihazlardaki tensor'lar işlem yapamaz
        print("─"*70)
        print("🔴 HATA: Farklı Cihazlardaki Tensor'lar")
        
        try:
            # YANLIŞ: CPU ve GPU tensor'ları toplanamaz
            result = cpu_tensor + gpu_tensor_1
        except RuntimeError as e:
            print(f"   ❌ HATA: {e}")
            print(f"   💡 ÇÖZÜM: Her iki tensor'u da aynı cihaza taşı")
            result = cpu_tensor.to('cuda') + gpu_tensor_1
            print(f"   ✅ Doğru: cpu_tensor.to('cuda') + gpu_tensor_1")
    else:
        print("⚠️  CUDA yok, bu bölüm atlanıyor.")


def demonstrate_performance_comparison() -> None:
    """
    CPU vs GPU performans karşılaştırması yapar.
    """
    print("\n" + "🎯 BÖLÜM 3: PERFORMANS KARŞILAŞTIRMASI - CPU VS GPU".center(70, "━"))
    
    sizes = [100, 500, 1000, 2000, 4000]
    
    print(f"\n{'Size':>6} | {'CPU (ms)':>10} | {'GPU (ms)':>10} | {'Speedup':>10}")
    print("─"*50)
    
    for size in sizes:
        # CPU matris çarpımı
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start = time.time()
        c_cpu = a_cpu @ b_cpu
        cpu_time = (time.time() - start) * 1000
        
        if torch.cuda.is_available():
            # GPU matris çarpımı
            a_gpu = a_cpu.to('cuda')
            b_gpu = b_cpu.to('cuda')
            
            # Warm-up (GPU'yu ısıt)
            _ = a_gpu @ b_gpu
            torch.cuda.synchronize()  # GPU işlemlerini bekle
            
            start = time.time()
            c_gpu = a_gpu @ b_gpu
            torch.cuda.synchronize()
            gpu_time = (time.time() - start) * 1000
            
            speedup = cpu_time / gpu_time
            print(f"{size:>6} | {cpu_time:>10.4f} | {gpu_time:>10.4f} | {speedup:>10.2f}x")
        else:
            print(f"{size:>6} | {cpu_time:>10.4f} | {'N/A':>10} | {'N/A':>10}")
    
    if torch.cuda.is_available():
        print(f"\n💡 Büyük matrisler için GPU {speedup:.0f}x daha hızlı!")
    else:
        print(f"\n⚠️  GPU yok, karşılaştırma yapılamadı.")


def demonstrate_memory_transfer_bottleneck() -> None:
    """
    CPU-GPU veri transferinin performans darboğazı olduğunu gösterir.
    """
    print("\n" + "🎯 BÖLÜM 4: BELLEK TRANSFER DARBOĞAZI - CPU ↔ GPU".center(70, "━"))
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA yok, bu bölüm atlanıyor.")
        return
    
    size = 4096
    
    # Senaryo 1: Her iterasyonda CPU → GPU transfer (KÖTÜ)
    print("🔴 KÖTÜ PRATİK: Her İterasyonda Transfer")
    
    total_time = 0
    for i in range(10):
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start = time.time()
        a_gpu = a_cpu.to('cuda')  # Transfer!
        b_gpu = b_cpu.to('cuda')  # Transfer!
        c_gpu = a_gpu @ b_gpu
        c_cpu = c_gpu.cpu()       # Transfer!
        torch.cuda.synchronize()
        total_time += time.time() - start
    
    bad_time = total_time * 1000
    print(f"   10 iterasyon: {bad_time:.2f} ms")
    print(f"   Her iterasyon: {bad_time/10:.2f} ms\n")
    
    # Senaryo 2: Veriyi GPU'da tut (İYİ)
    print("─"*70)
    print("✅ İYİ PRATİK: Veriyi GPU'da Tut")
    
    a_gpu = torch.randn(size, size, device='cuda')  # Doğrudan GPU'da oluştur
    b_gpu = torch.randn(size, size, device='cuda')
    
    torch.cuda.synchronize()
    start = time.time()
    for i in range(10):
        c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    good_time = (time.time() - start) * 1000
    
    print(f"   10 iterasyon: {good_time:.2f} ms")
    print(f"   Her iterasyon: {good_time/10:.2f} ms")
    
    speedup = bad_time / good_time
    print(f"\n🚀 HIZ ARTIŞI: {speedup:.1f}x daha hızlı!")
    print(f"💡 Sebep: CPU-GPU transfer overhead'i yok")


def demonstrate_pinned_memory() -> None:
    """
    Pinned memory (page-locked memory) kullanımını gösterir.
    """
    print("\n" + "🎯 BÖLÜM 5: PINNED MEMORY - HIZLI TRANSFER".center(70, "━"))
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA yok, bu bölüm atlanıyor.")
        return
    
    size = (1000, 1000)
    
    # Normal CPU tensor
    print("🔹 Normal CPU Tensor → GPU Transfer")
    
    normal_tensor = torch.randn(*size)
    
    start = time.time()
    for _ in range(100):
        _ = normal_tensor.to('cuda')
    torch.cuda.synchronize()
    normal_time = (time.time() - start) * 1000
    
    print(f"   100 transfer: {normal_time:.2f} ms\n")
    
    # Pinned memory tensor
    print("─"*70)
    print("🔹 Pinned Memory Tensor → GPU Transfer")
    
    pinned_tensor = torch.randn(*size).pin_memory()
    
    start = time.time()
    for _ in range(100):
        _ = pinned_tensor.to('cuda', non_blocking=True)
    torch.cuda.synchronize()
    pinned_time = (time.time() - start) * 1000
    
    print(f"   100 transfer: {pinned_time:.2f} ms")
    
    speedup = normal_time / pinned_time
    print(f"\n🚀 HIZ ARTIŞI: {speedup:.2f}x daha hızlı!")
    print(f"💡 Pinned memory, DMA (Direct Memory Access) kullanır")
    print(f"⚠️  Dikkat: Pinned memory sistem RAM'ini kilitler, fazla kullanma!")


def demonstrate_cuda_streams() -> None:
    """
    CUDA streams ile paralel işlem yapmayı gösterir.
    """
    print("\n" + "🎯 BÖLÜM 6: CUDA STREAMS - PARALEL İŞLEM".center(70, "━"))
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA yok, bu bölüm atlanıyor.")
        return
    
    size = 2048
    
    # Senaryo 1: Sıralı işlem (default stream)
    print("🔹 Sıralı İşlem (Default Stream)")
    
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    c = torch.randn(size, size, device='cuda')
    
    torch.cuda.synchronize()
    start = time.time()
    
    result1 = a @ b
    result2 = b @ c
    result3 = a @ c
    
    torch.cuda.synchronize()
    sequential_time = (time.time() - start) * 1000
    
    print(f"   3 matris çarpımı: {sequential_time:.2f} ms\n")
    
    # Senaryo 2: Paralel işlem (multiple streams)
    print("─"*70)
    print("🔹 Paralel İşlem (Multiple Streams)")
    
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    stream3 = torch.cuda.Stream()
    
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.cuda.stream(stream1):
        result1 = a @ b
    
    with torch.cuda.stream(stream2):
        result2 = b @ c
    
    with torch.cuda.stream(stream3):
        result3 = a @ c
    
    torch.cuda.synchronize()
    parallel_time = (time.time() - start) * 1000
    
    print(f"   3 matris çarpımı: {parallel_time:.2f} ms")
    
    speedup = sequential_time / parallel_time
    print(f"\n🚀 HIZ ARTIŞI: {speedup:.2f}x daha hızlı!")
    print(f"💡 Bağımsız işlemler paralel çalıştırıldı")


def demonstrate_memory_management() -> None:
    """
    GPU bellek yönetimi ve optimizasyon tekniklerini gösterir.
    """
    print("\n" + "🎯 BÖLÜM 7: GPU BELLEK YÖNETİMİ".center(70, "━"))
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA yok, bu bölüm atlanıyor.")
        return
    
    # Bellek durumunu göster
    print("🔹 GPU Bellek Durumu")
    
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    
    print(f"   Allocated: {allocated:.2f} MB")
    print(f"   Reserved: {reserved:.2f} MB\n")
    
    # Büyük tensor oluştur
    print("─"*70)
    print("🔹 Büyük Tensor Oluşturma")
    
    big_tensor = torch.randn(10000, 10000, device='cuda')
    
    allocated_after = torch.cuda.memory_allocated() / 1024**2
    print(f"   Tensor boyutu: {big_tensor.element_size() * big_tensor.nelement() / 1024**2:.2f} MB")
    print(f"   Allocated: {allocated_after:.2f} MB (+{allocated_after - allocated:.2f} MB)\n")
    
    # Belleği temizle
    print("─"*70)
    print("🔹 Bellek Temizleme")
    
    del big_tensor
    torch.cuda.empty_cache()
    
    allocated_cleaned = torch.cuda.memory_allocated() / 1024**2
    reserved_cleaned = torch.cuda.memory_reserved() / 1024**2
    
    print(f"   Allocated: {allocated_cleaned:.2f} MB")
    print(f"   Reserved: {reserved_cleaned:.2f} MB")
    print(f"   💡 empty_cache() reserved memory'yi serbest bıraktı\n")
    
    # Bellek profiling
    print("─"*70)
    print("🔹 Bellek Profiling")
    
    print(f"   Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print(f"   Max reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
    
    # Reset statistics
    torch.cuda.reset_peak_memory_stats()
    print(f"   💡 reset_peak_memory_stats() ile istatistikler sıfırlandı")


def demonstrate_common_pitfalls() -> None:
    """
    GPU kullanımında sık yapılan hataları gösterir.
    """
    print("\n" + "🎯 BÖLÜM 8: YAYGIN HATALAR VE ÇÖZÜMLER".center(70, "━"))
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA yok, bu bölüm atlanıyor.")
        return
    
    # HATA 1: synchronize() unutmak
    print("🔴 HATA 1: torch.cuda.synchronize() Unutmak")
    
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.randn(1000, 1000, device='cuda')
    
    # YANLIŞ: GPU işlemi asenkron, zaman ölçümü yanlış
    start = time.time()
    c = a @ b
    wrong_time = (time.time() - start) * 1000
    
    # DOĞRU: synchronize() ile bekle
    start = time.time()
    c = a @ b
    torch.cuda.synchronize()
    correct_time = (time.time() - start) * 1000
    
    print(f"   Synchronize olmadan: {wrong_time:.6f} ms (YANLIŞ!)")
    print(f"   Synchronize ile: {correct_time:.4f} ms (DOĞRU)")
    print(f"   💡 GPU işlemleri asenkron, mutlaka synchronize() kullan!\n")
    
    # HATA 2: Gereksiz CPU-GPU transfer
    print("─"*70)
    print("🔴 HATA 2: Gereksiz CPU-GPU Transfer")
    
    # YANLIŞ: Her iterasyonda .item() çağırma
    loss_values = []
    tensor = torch.randn(1000, device='cuda')
    
    start = time.time()
    for i in range(1000):
        loss = tensor.sum()
        loss_values.append(loss.item())  # CPU'ya transfer!
    wrong_time = (time.time() - start) * 1000
    
    # DOĞRU: GPU'da topla, sonra bir kez transfer et
    start = time.time()
    losses = []
    for i in range(1000):
        loss = tensor.sum()
        losses.append(loss)
    
    loss_values = [l.item() for l in losses]  # Tek seferde
    correct_time = (time.time() - start) * 1000
    
    print(f"   Her iterasyonda .item(): {wrong_time:.2f} ms")
    print(f"   Sonunda toplu .item(): {correct_time:.2f} ms")
    print(f"   🚀 {wrong_time/correct_time:.1f}x daha hızlı!")


def main() -> None:
    """
    Ana çalıştırma fonksiyonu.
    """
    print("\n" + "="*70)
    print("🚀 GPU HIZLANDIRMA - CUDA VE PERFORMANS OPTİMİZASYONU".center(70))
    print("="*70)
    
    check_cuda_availability()
    demonstrate_device_management()
    demonstrate_performance_comparison()
    demonstrate_memory_transfer_bottleneck()
    demonstrate_pinned_memory()
    demonstrate_cuda_streams()
    demonstrate_memory_management()
    demonstrate_common_pitfalls()
    
    print("\n" + "="*70)
    print("✅ DERS 05 TAMAMLANDI!".center(70))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
