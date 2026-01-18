# 🎬 DERS 01: TENSOR MEKANİĞİ - BELLEK DÜZENİ VE STRIDE ANALİZİ

---

## 📺 BLOK 1: PRODÜKSİYON VE SENARYO (YouTuber Modu)

### 🎯 Video Başlığı
**"PyTorch Tensor'ları Neden NumPy'dan Hızlı? | Stride ve Storage Mekanizması Deep Dive"**

### 🎣 The Hook (0:00-0:45)
> "Çoğu kişi PyTorch tensor'larını sadece 'GPU destekli NumPy' sanıyor. Ama gerçek şu: Bir tensor'u transpose ettiğinizde bellekte TEK BİR BYTE bile hareket etmiyor! Peki PyTorch bunu nasıl yapıyor? Bugün tensor'ların bellek anatomisine gireceğiz ve `.view()` ile `.reshape()` arasındaki farkı öğrenince production'da karşılaştığınız memory leak'lerin %80'ini çözeceksiniz. Hadi başlayalım!"

### 🎨 Görselleştirme İpuçları
1. **0:45-1:30**: Ekrana bir 2D matris göster. Transpose butonuna basıldığında, bellekteki veri bloğu AYNI kalsın ama üzerindeki "okuma yönü okları" 90 derece dönsün. Yanında "Zero Copy!" yazısı belirsin.

2. **3:00-4:00**: Stride animasyonu: 3x4'lük bir matris göster. `matrix[1,2]` elemanına erişirken, bellek bloğunda "base + 1×4 + 2×1 = 6" hesaplamasını adım adım animasyonla göster.

3. **6:30-7:15**: Storage paylaşımı: İki farklı tensor (orijinal ve sliced) göster. Altlarında TEK BİR ortak storage bloğu olsun. Her tensor'un farklı offset'ten başladığını renkli ok ile göster.

4. **10:00-11:00**: Contiguous vs Non-contiguous karşılaştırması: İki bellek bloğu yan yana. Birinde elemanlar sıralı (yeşil), diğerinde atlayarak okunuyor (kırmızı kesik çizgiler).

---

## 🧠 BLOK 3: DERİN TEORİK ANALİZ (Akademisyen Modu)

### 📐 Matematiksel Temeller

#### 1. Stride Hesaplama Formülü
Bir N-boyutlu tensor için `i,j,k,...` indeksindeki elemana erişim:

```
memory_address = base_pointer + (i × stride[0]) + (j × stride[1]) + (k × stride[2]) + ...
```

**Örnek:** `tensor.shape = (3, 4)` için:
- `stride = (4, 1)` → Row-major order (C-style)
- `tensor[1, 2]` → `base + 1×4 + 2×1 = base + 6`

**Transpose sonrası:**
- `stride = (1, 4)` → Column-major order
- Bellekte veri değişmedi, sadece stride değişti!

#### 2. Storage Offset Matematiği
Bir tensor'u slice ettiğinizde:

```python
original = torch.arange(12)  # storage: [0,1,2,...,11]
sliced = original[3:9]       # storage_offset = 3
```

`sliced[0]` → `original.storage()[3]` (Aynı bellek!)

---

### ⚙️ Under The Hood (Kaputun Altı)

#### PyTorch C++ Katmanında Neler Oluyor?

**1. TensorImpl Sınıfı (C++)**
PyTorch'un Python API'si altında `c10::TensorImpl` sınıfı çalışır:

```cpp
class TensorImpl {
  Storage storage_;           // Ham veri (1D array)
  int64_t storage_offset_;    // Başlangıç noktası
  SmallVector<int64_t> sizes_;    // Shape bilgisi
  SmallVector<int64_t> strides_;  // Adım bilgisi
  // ...
};
```

**2. View İşlemi (Zero-Copy)**
`.view()` çağrıldığında:
- Yeni bir `TensorImpl` oluşturulur
- `storage_` pointer'ı KOPYALANMAZ (referans paylaşılır)
- Sadece `sizes_` ve `strides_` yeniden hesaplanır
- **Maliyet:** O(1) - Sabit zaman!

**3. Contiguous Kontrolü**
PyTorch, bir tensor'un contiguous olup olmadığını şu şekilde kontrol eder:

```cpp
bool is_contiguous() {
  int64_t expected_stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    if (stride[i] != expected_stride) return false;
    expected_stride *= size[i];
  }
  return true;
}
```

**4. CUDA Kernel Optimizasyonu**
Contiguous tensor'lar GPU'da **coalesced memory access** sağlar:
- Warp içindeki 32 thread bitişik adresleri okur → Tek memory transaction
- Non-contiguous tensor → Her thread farklı adresten okur → 32 ayrı transaction!
- **Performans farkı:** 10x-100x hız kaybı olabilir

---

### 🏭 Sektör Notu: Production Ortamında Karşılaşılan Sorunlar

#### Problem 1: Memory Leak (Bellek Sızıntısı)
**Senaryo:** Büyük bir model'den sürekli `.view()` ile küçük tensor'lar çıkarıyorsunuz.

```python
# YANLIŞ KULLANIM
big_tensor = torch.randn(10000, 10000)  # 400 MB
for i in range(1000):
    small = big_tensor[i:i+10].view(-1)
    process(small)
# big_tensor hala bellekte! Çünkü small'lar storage'ı referans ediyor.
```

**Çözüm:**
```python
small = big_tensor[i:i+10].clone()  # Yeni storage oluştur
```

#### Problem 2: ONNX Export Hatası
**Senaryo:** Model'inizi ONNX'e export ederken "RuntimeError: view size is not compatible" hatası.

**Sebep:** Non-contiguous tensor'da `.view()` kullanımı.

**Çözüm:**
```python
# Model içinde
x = x.permute(0, 2, 1)  # Non-contiguous hale gelir
x = x.contiguous()      # ONNX için zorunlu!
x = x.view(batch, -1)
```

#### Problem 3: Mobile Deployment (TorchScript)
**Senaryo:** `torch.jit.trace()` ile model export edilirken stride bilgisi kaybolur.

**Çözüm:** Tüm tensor'ları `.contiguous()` ile işaretle:
```python
@torch.jit.script
def forward(x):
    x = x.contiguous()  # Mobile için garanti
    return model(x)
```

---

### 📊 Performans Karşılaştırması

| İşlem | Bellek Kopyalama | Zaman Karmaşıklığı | GPU Uyumluluğu |
|-------|------------------|-------------------|----------------|
| `.view()` | ❌ Hayır (Zero-copy) | O(1) | ⚠️ Contiguous gerekir |
| `.reshape()` | ⚠️ Gerekirse | O(1) veya O(n) | ✅ Her zaman |
| `.clone()` | ✅ Evet | O(n) | ✅ Her zaman |
| `.contiguous()` | ⚠️ Gerekirse | O(1) veya O(n) | ✅ Her zaman |

---

### 🔬 Derin Dalış: Transpose Neden Bu Kadar Hızlı?

**Soru:** 1 milyar elemanlı bir matrisi transpose etmek neden 0.001 saniye sürüyor?

**Cevap:** Çünkü hiçbir veri hareket etmiyor!

```python
import torch
import time

big = torch.randn(10000, 10000)  # 100 milyon eleman

start = time.time()
transposed = big.t()
print(f"Transpose süresi: {time.time() - start:.6f} saniye")
# Çıktı: ~0.000050 saniye (50 mikrosaniye!)

# Doğrulama
print(f"Aynı storage? {big.data_ptr() == transposed.data_ptr()}")  # True
```

**Açıklama:**
- `big.stride() = (10000, 1)` → Satır öncelikli
- `transposed.stride() = (1, 10000)` → Sütun öncelikli
- Bellekte veri: `[0,1,2,3,...,99999999]` (DEĞİŞMEDİ!)
- Sadece metadata güncellendi (stride ve shape)

---

### 🎓 Akademik Referanslar

1. **Tensor Storage Model:** 
   - "Automatic differentiation in PyTorch" (Paszke et al., 2017)
   - Section 3.2: Storage and View Mechanism

2. **Memory Layout Optimization:**
   - "Halide: A Language for Fast, Portable Computation on Images" (Ragan-Kelley et al., 2013)
   - Stride-based memory access patterns

3. **CUDA Coalesced Access:**
   - NVIDIA CUDA C Programming Guide, Section 5.3.2
   - "Global Memory Coalescing"

---

## ⚔️ BLOK 4: MEYDAN OKUMA (Ödev)

### 🎯 Görev: NumPy ile Stride Mekanizmasını Yeniden Yaz

**Zorluk Seviyesi:** 🔥🔥🔥 (Orta-İleri)

**Açıklama:**
PyTorch'un `view()` ve `transpose()` işlemlerini **sadece NumPy kullanarak** ve **hiçbir veri kopyalamadan** implemente edin.

**Gereksinimler:**

```python
import numpy as np

class CustomTensor:
    def __init__(self, data: np.ndarray):
        """
        data: 1D NumPy array (storage)
        """
        # TODO: storage, shape, stride, offset değişkenlerini tanımla
        pass
    
    def view(self, *new_shape):
        """
        PyTorch'un .view() metodunu taklit et.
        Yeni bir CustomTensor döndür (storage paylaşımlı).
        """
        # TODO: Yeni stride hesapla, storage'ı paylaş
        pass
    
    def transpose(self, dim0, dim1):
        """
        İki boyutu yer değiştir (stride manipülasyonu).
        """
        # TODO: Stride'ı değiştir, veri kopyalama!
        pass
    
    def __getitem__(self, index):
        """
        Stride kullanarak doğru elemana eriş.
        """
        # TODO: Stride formülünü uygula
        pass
    
    def is_contiguous(self) -> bool:
        """
        Tensor'un contiguous olup olmadığını kontrol et.
        """
        # TODO: Stride sırasını kontrol et
        pass

# Test kodu
storage = np.arange(12, dtype=np.float32)
tensor = CustomTensor(storage)
tensor = tensor.view(3, 4)
print(tensor[1, 2])  # Beklenen: 6.0

transposed = tensor.transpose(0, 1)
print(transposed.is_contiguous())  # Beklenen: False
```

**Bonus Görev:**
- `contiguous()` metodunu ekle (gerekirse veriyi yeniden düzenle)
- `__repr__()` ile tensor'u güzel yazdır
- PyTorch ile sonuçları karşılaştır ve doğrula

**Teslim:**
- GitHub Gist linki veya `.py` dosyası
- Test sonuçlarını içeren ekran görüntüsü

---

### ✅ Başarı Kriterleri
1. ✅ Hiçbir `np.reshape()` veya `np.transpose()` kullanmadınız mı?
2. ✅ Storage'ı kopyalamadan view oluşturabildiniz mi?
3. ✅ Stride formülü doğru çalışıyor mu?
4. ✅ PyTorch sonuçlarıyla %100 eşleşiyor mu?

---

## 📚 Ek Kaynaklar

- [PyTorch Internals - Tensor Storage](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [Stride Tricks in NumPy](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html)
- [CUDA Memory Coalescing](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)

---

**🎬 Sonraki Ders:** `02_tensor_math_gemm.py` - Matris Çarpımı ve GEMM Optimizasyonu
