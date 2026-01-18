# 🎬 DERS 02: TENSOR MATEMATİĞİ - GEMM VE BROADCASTING

---

## 📺 BLOK 1: PRODÜKSİYON VE SENARYO (YouTuber Modu)

### 🎯 Video Başlığı
**"GEMM Nedir ve Neden Her Deep Learning Framework'ün Kalbi? | Broadcasting Deep Dive"**

### 🎣 The Hook (0:00-0:45)
> "Bilgisayar biliminde en optimize edilmiş algoritma hangisidir? Sorting? Hayır. Search? Hayır. GEMM - General Matrix Multiply! Tüm derin öğrenme modellerinin %90'ı aslında GEMM çağrısıdır. NVIDIA'nın milyarlarca dolarlık GPU'ları sadece bu işlemi hızlandırmak için tasarlanmıştır. Bugün PyTorch'un matris çarpımını nasıl 1000x hızlandırdığını ve broadcasting'in sihirli kurallarını öğreneceksiniz. Hadi başlayalım!"

### 🎨 Görselleştirme İpuçları

1. **1:00-2:00**: GEMM Animasyonu
   - Ekrana iki matris (A: 3x4, B: 4x5) göster
   - C[0,0] hesaplanırken: A'nın 0. satırı ile B'nin 0. sütunu element-wise çarpılıp toplanırken animasyon göster
   - Her adımda çarpılan elemanlar yanıp sönsün
   - Sonuç: "1×5 + 2×9 + 3×13 + 4×17 = 110"

2. **4:30-5:30**: Naive vs Optimized Karşılaştırması
   - Split screen: Solda 3 iç içe döngü (yavaş), sağda BLAS kütüphanesi (hızlı)
   - Solda döngüler dönüyor (yavaş animasyon), sağda tek seferde "BOOM!" sonuç çıkıyor
   - Altında hız karşılaştırması: "128x128 matris → Naive: 2.5s, BLAS: 0.002s (1250x hızlı!)"

3. **7:00-9:00**: Broadcasting Kuralları
   - Ekrana iki tensor göster: (3, 1, 5) ve (1, 4, 5)
   - Sağdan sola boyutları karşılaştır (animasyonla)
   - Uyumlu boyutlar yeşil, uyumsuz kırmızı işaretle
   - Sonuç tensor'u (3, 4, 5) şeklinde genişlerken göster

4. **11:00-12:00**: Vectorization Gücü
   - 1 milyon elemanlı iki vektör göster
   - Python loop: Her eleman tek tek toplanıyor (yavaş)
   - SIMD: 8 eleman aynı anda toplanıyor (hızlı)
   - CPU register'larında paralel işlem animasyonu

---

## 🧠 BLOK 3: DERİN TEORİK ANALİZ (Akademisyen Modu)

### 📐 Matematiksel Temeller

#### 1. GEMM Formülü
**General Matrix Multiply (GEMM):**

```
C = α × (A @ B) + β × C

Burada:
- A: (m × k) matrisi
- B: (k × n) matrisi
- C: (m × n) matrisi
- α, β: Skaler katsayılar
```

**Element-wise açılım:**
```
C[i, j] = α × Σ(k=0 to K-1) A[i, k] × B[k, j] + β × C[i, j]
```

**Karmaşıklık Analizi:**
- **Zaman:** O(m × n × k) → 3 iç içe döngü
- **Bellek:** O(m×k + k×n + m×n)
- **FLOP Count:** 2mnk (Her eleman için k çarpma + k toplama)

**Örnek:** (1000 × 1000) @ (1000 × 1000)
- FLOP: 2 × 1000³ = 2 milyar işlem
- Modern GPU (A100): ~312 TFLOPS → ~6.4 mikrosaniye!

---

#### 2. Broadcasting Kuralları (Formal Tanım)

**Kural 1:** Sağdan sola boyutları karşılaştır
```python
A: (5, 3, 4, 1)
B:    (3, 1, 7)
─────────────────
Result: (5, 3, 4, 7)
```

**Kural 2:** İki boyut uyumludur ⟺ (eşit VEYA birisi 1)
```
Uyumlu:
  3 vs 3 ✅
  3 vs 1 ✅
  1 vs 7 ✅

Uyumsuz:
  3 vs 5 ❌ (İkisi de 1 değil ve eşit değil)
```

**Kural 3:** Eksik boyutlar 1 kabul edilir
```python
A: (4, 5)    →  (1, 4, 5)  # Sol tarafa 1 eklenir
B: (5,)      →  (1, 1, 5)
```

**Matematiksel Gösterim:**
```
Broadcast(A, B) = C
where C[i₁, i₂, ..., iₙ] = A[j₁, j₂, ..., jₘ] ⊙ B[k₁, k₂, ..., kₚ]

jₓ = iₓ if A.shape[x] > 1 else 0
kₓ = iₓ if B.shape[x] > 1 else 0
```

---

### ⚙️ Under The Hood (Kaputun Altı)

#### BLAS/cuBLAS Kütüphaneleri

**1. CPU: BLAS (Basic Linear Algebra Subprograms)**

PyTorch CPU'da matris çarpımı için Intel MKL veya OpenBLAS kullanır:

```cpp
// PyTorch C++ backend
// aten/src/ATen/native/LinearAlgebra.cpp

Tensor matmul_cpu(const Tensor& A, const Tensor& B) {
  // Intel MKL'nin SGEMM fonksiyonunu çağır
  cblas_sgemm(
    CblasRowMajor,    // Row-major order
    CblasNoTrans,     // A transpose edilmemiş
    CblasNoTrans,     // B transpose edilmemiş
    m, n, k,          // Boyutlar
    1.0,              // alpha
    A.data_ptr(),     // A pointer
    lda,              // Leading dimension
    B.data_ptr(),     // B pointer
    ldb,
    0.0,              // beta
    C.data_ptr(),     // C pointer
    ldc
  );
}
```

**Intel MKL Optimizasyonları:**
- **Cache Blocking:** Matrisi küçük bloklara böl, L1/L2 cache'e sığdır
- **Loop Unrolling:** Döngü overhead'ini azalt
- **SIMD (AVX-512):** 16 float'u aynı anda işle
- **Multi-threading:** OpenMP ile paralel hesaplama

---

**2. GPU: cuBLAS (CUDA BLAS)**

GPU'da NVIDIA'nın cuBLAS kütüphanesi kullanılır:

```cpp
// PyTorch CUDA backend
// aten/src/ATen/native/cuda/Blas.cpp

Tensor matmul_cuda(const Tensor& A, const Tensor& B) {
  cublasHandle_t handle = getCurrentCUDABlasHandle();
  
  // cuBLAS SGEMM çağrısı
  cublasSgemm(
    handle,
    CUBLAS_OP_N,      // B transpose edilmemiş
    CUBLAS_OP_N,      // A transpose edilmemiş
    n, m, k,
    &alpha,
    B.data_ptr(),     // cuBLAS column-major kullanır!
    ldb,
    A.data_ptr(),
    lda,
    &beta,
    C.data_ptr(),
    ldc
  );
}
```

**cuBLAS Optimizasyonları:**
- **Tensor Cores (A100):** 4×4 matris bloklarını tek cycle'da çarp
- **Warp-level Primitives:** 32 thread aynı anda çalışır
- **Shared Memory:** On-chip bellek (L1 cache benzeri)
- **Kernel Fusion:** Birden fazla işlemi tek kernel'da birleştir

---

#### GEMM Performans Analizi

**Roofline Model:**
```
Performans = min(Peak FLOPS, Bandwidth × Arithmetic Intensity)

Arithmetic Intensity (AI) = FLOP / Byte
GEMM AI = 2mnk / (4(mk + kn + mn))  # float32 için 4 byte

Örnek: (1024 × 1024) @ (1024 × 1024)
AI = 2×1024³ / (4×3×1024²) ≈ 170 FLOP/Byte
→ Compute-bound (Bellek değil, hesaplama sınırlı)
```

**GPU Kullanım Oranı:**
```python
import torch

A = torch.randn(4096, 4096, device='cuda')
B = torch.randn(4096, 4096, device='cuda')

# Profiling
with torch.profiler.profile() as prof:
    C = A @ B

print(prof.key_averages().table())
# Çıktı: ~95% GPU kullanımı (Çok iyi!)
```

---

### 🏭 Sektör Notu: Production Ortamında Karşılaşılan Sorunlar

#### Problem 1: Mixed Precision Training'de Overflow

**Senaryo:** FP16 (half precision) kullanırken matris çarpımında overflow.

```python
# YANLIŞ
A = torch.randn(1000, 1000, dtype=torch.float16, device='cuda')
B = torch.randn(1000, 1000, dtype=torch.float16, device='cuda')
C = A @ B  # Overflow riski! FP16 max: 65504
```

**Çözüm:** Automatic Mixed Precision (AMP)
```python
# DOĞRU
from torch.cuda.amp import autocast

with autocast():
    C = A @ B  # PyTorch otomatik FP32'ye yükseltir
```

---

#### Problem 2: Broadcasting ile Beklenmedik Bellek Kullanımı

**Senaryo:** Büyük tensor'lara broadcasting uygulanırken OOM (Out of Memory).

```python
# YANLIŞ
big_tensor = torch.randn(1000, 1000, 1000, device='cuda')  # 4 GB
small_tensor = torch.randn(1000, device='cuda')            # 4 KB

result = big_tensor + small_tensor  # small_tensor (1000, 1000, 1000)'e genişler!
# Geçici bellek: 4 GB ekstra → OOM!
```

**Çözüm:** In-place işlem
```python
# DOĞRU
big_tensor.add_(small_tensor)  # In-place, ekstra bellek yok
```

---

#### Problem 3: Batch Matmul'da Boyut Karışıklığı

**Senaryo:** Transformer'da attention hesaplaması.

```python
# YANLIŞ
Q = torch.randn(32, 8, 128, 64)  # (batch, heads, seq_len, d_k)
K = torch.randn(32, 8, 128, 64)

# Hata: Son iki boyut uyumsuz (64 @ 64)
scores = Q @ K  # RuntimeError!
```

**Çözüm:** Transpose
```python
# DOĞRU
scores = Q @ K.transpose(-2, -1)  # (32, 8, 128, 64) @ (32, 8, 64, 128)
# Sonuç: (32, 8, 128, 128) ✅
```

---

### 📊 Performans Karşılaştırması

| İşlem | CPU (Intel i9) | GPU (RTX 3090) | GPU (A100) | Hızlanma |
|-------|---------------|----------------|------------|----------|
| (1024×1024) @ (1024×1024) | 15 ms | 0.8 ms | 0.3 ms | 50x |
| (4096×4096) @ (4096×4096) | 980 ms | 12 ms | 4 ms | 245x |
| Batch (32, 512, 512) | 1200 ms | 18 ms | 6 ms | 200x |

**Not:** A100'ün Tensor Core'ları FP16'da 312 TFLOPS ulaşır!

---

### 🔬 Derin Dalış: Tensor Core Mimarisi

**NVIDIA Tensor Core (A100):**
```
Bir Tensor Core cycle'da şunu yapar:
D = A × B + C

Burada:
- A: 4×4 matris (FP16)
- B: 4×4 matris (FP16)
- C: 4×4 matris (FP32)
- D: 4×4 matris (FP32)

Toplam: 64 FLOP (4×4×4 çarpma + 4×4 toplama) tek cycle'da!
```

**Kullanım:**
```python
# PyTorch otomatik Tensor Core kullanır (FP16 + AMP)
with torch.cuda.amp.autocast():
    C = A @ B  # Tensor Core aktif!
```

---

## ⚔️ BLOK 4: MEYDAN OKUMA (Ödev)

### 🎯 Görev: Cache-Optimized GEMM Implementasyonu

**Zorluk Seviyesi:** 🔥🔥🔥🔥 (İleri)

**Açıklama:**
Naive GEMM'den daha hızlı bir implementasyon yazın. Cache blocking tekniğini kullanarak Intel MKL'ye yakın performans elde edin.

**Gereksinimler:**

```python
import torch
import time

def blocked_matmul(A: torch.Tensor, B: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """
    Cache-aware matris çarpımı.
    
    Args:
        A: (m, k) tensor
        B: (k, n) tensor
        block_size: Cache'e sığacak blok boyutu
    
    Returns:
        C: (m, n) tensor
    """
    m, k = A.shape
    k2, n = B.shape
    assert k == k2
    
    C = torch.zeros(m, n, dtype=A.dtype)
    
    # TODO: Matrisleri block_size × block_size bloklara böl
    # TODO: Her blok çiftini çarp (cache'de kal)
    # TODO: Sonuçları C'ye akümüle et
    
    return C

# Test
sizes = [128, 256, 512, 1024]
for size in sizes:
    A = torch.randn(size, size)
    B = torch.randn(size, size)
    
    # Naive
    start = time.time()
    C_naive = naive_matmul(A, B)  # Ders 02'deki fonksiyon
    naive_time = time.time() - start
    
    # Blocked
    start = time.time()
    C_blocked = blocked_matmul(A, B, block_size=64)
    blocked_time = time.time() - start
    
    # PyTorch
    start = time.time()
    C_torch = A @ B
    torch_time = time.time() - start
    
    print(f"\nSize: {size}×{size}")
    print(f"Naive:   {naive_time:.4f}s")
    print(f"Blocked: {blocked_time:.4f}s (Speedup: {naive_time/blocked_time:.1f}x)")
    print(f"PyTorch: {torch_time:.6f}s")
    
    # Doğruluk kontrolü
    assert torch.allclose(C_blocked, C_torch, atol=1e-4)
```

**Bonus Görevler:**
1. **Loop Unrolling:** İç döngüyü 4'lü gruplar halinde aç
2. **SIMD Simulation:** `torch.sum()` yerine manuel toplama yap
3. **Profiling:** Hangi blok boyutu en hızlı? (16, 32, 64, 128 dene)
4. **Visualization:** Blok boyutuna göre performans grafiği çiz

**Beklenen Sonuç:**
- Naive'den en az 5-10x hızlı
- PyTorch'tan yavaş ama yakın (2-5x fark kabul edilebilir)

---

### ✅ Başarı Kriterleri
1. ✅ Cache blocking doğru uygulandı mı?
2. ✅ Naive implementasyondan en az 5x hızlı mı?
3. ✅ Sonuçlar PyTorch ile eşleşiyor mu? (atol=1e-4)
4. ✅ Farklı blok boyutlarını test ettiniz mi?

---

## 📚 Ek Kaynaklar

- [BLAS (Basic Linear Algebra Subprograms)](http://www.netlib.org/blas/)
- [Intel MKL Documentation](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html)
- [NVIDIA cuBLAS](https://docs.nvidia.com/cuda/cublas/)
- [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
- [PyTorch Broadcasting Semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)

---

**🎬 Sonraki Ders:** `03_indexing_advanced.py` - Masking, Fancy Indexing ve View vs Copy
