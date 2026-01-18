# 🔧 FAZ 1: TENSORS & COMPUTATIONAL GRAPH (THE ENGINE)

## ✅ Tamamlandı!

Bu faz, PyTorch'un temelini oluşturan **tensor mekanikleri** ve **otomatik türev sistemi**ni kapsar. "Under the Hood" seviyesinde, bellekten GPU'ya, stride'dan autograd'a kadar her şeyi öğrendiniz.

---

## 📚 Dersler

| # | Dosya | Konu | Durum |
|---|-------|------|-------|
| **01** | `01_tensor_mechanics.py` | Tensor vs NumPy, Storage, Offset, Stride | ✅ |
| **02** | `02_tensor_math_gemm.py` | GEMM, Broadcasting, Vectorization | ✅ |
| **03** | `03_indexing_advanced.py` | Masking, Fancy Indexing, View vs Copy | ✅ |
| **04** | `04_manipulation_view_reshape.py` | view(), reshape(), permute(), transpose() | ✅ |
| **05** | `05_gpu_acceleration.py` | CUDA, CPU-GPU Transfer, Bottleneck | ✅ |
| **06** | `06_autograd_engine.py` | DAG, .backward(), Gradient Flow | ✅ |
| **07** | `07_custom_autograd.py` | torch.autograd.Function, Custom Derivatives | ✅ |

---

## 🎯 Öğrendikleriniz

### 🧠 Kavramsal Bilgi
- ✅ **Tensor Anatomy:** Storage, Offset, Stride, Contiguous Memory
- ✅ **GEMM Optimization:** BLAS/cuBLAS, Cache Blocking, Tensor Cores
- ✅ **Broadcasting Rules:** Otomatik boyut genişletme mekanizması
- ✅ **View vs Copy:** Bellek paylaşımı ve optimizasyon
- ✅ **GPU Programming:** CUDA kernels, Streams, Pinned Memory
- ✅ **Autograd Engine:** DAG yapısı, Backward propagation
- ✅ **Custom Gradients:** torch.autograd.Function ile özel türevler

### 💻 Pratik Beceriler
- ✅ Tensor'ların bellekte nasıl yerleştiğini analiz etme
- ✅ CPU-GPU transfer darboğazlarını tespit etme ve çözme
- ✅ Gradient flow'u debug etme
- ✅ Kendi activation function'larınızı yazma
- ✅ Numerical gradient checking ile doğrulama

### 🏭 Production Bilgisi
- ✅ Memory leak'leri önleme
- ✅ ONNX export sorunlarını çözme
- ✅ Mixed precision training (FP16/FP32)
- ✅ Gradient accumulation stratejileri
- ✅ Inference optimization (no_grad vs inference_mode)

---

## 🚀 Hızlı Test

Tüm dersleri çalıştırın:

```bash
cd Faz_1_Tensors

# Her dersi sırayla çalıştır
python 01_tensor_mechanics.py
python 02_tensor_math_gemm.py
python 03_indexing_advanced.py
python 04_manipulation_view_reshape.py
python 05_gpu_acceleration.py
python 06_autograd_engine.py
python 07_custom_autograd.py
```

---

## 📊 Performans Karşılaştırmaları

Bu fazda öğrendiğiniz optimizasyonların etkisi:

| Optimizasyon | Hız Artışı | Bellek Tasarrufu |
|--------------|------------|------------------|
| View vs Clone | ∞ (Zero-copy) | %100 |
| GEMM (BLAS vs Naive) | 1000x+ | - |
| GPU vs CPU (Büyük matris) | 100-500x | - |
| Pinned Memory | 2-3x | - |
| CUDA Streams | 2-4x | - |
| no_grad() | 1.5-2x | %50 |
| Gradient Checkpointing | 0.7x (yavaş) | %50 |

---

## 🎓 Önemli Kavramlar

### 1️⃣ Contiguous Memory
```python
# Non-contiguous
x = torch.randn(3, 4)
y = x.t()  # Transpose
print(y.is_contiguous())  # False

# Contiguous yap
y = y.contiguous()  # Yeni bellek ayırır!
```

### 2️⃣ GPU Transfer Optimization
```python
# KÖTÜ: Her iterasyonda transfer
for i in range(1000):
    x_gpu = x_cpu.to('cuda')
    y = model(x_gpu)
    loss = y.cpu()

# İYİ: Veriyi GPU'da tut
x_gpu = x_cpu.to('cuda')
for i in range(1000):
    y = model(x_gpu)
```

### 3️⃣ Gradient Accumulation
```python
# YANLIŞ: Gradientler birikiyor
for epoch in range(10):
    loss = model(x)
    loss.backward()
    optimizer.step()

# DOĞRU: Her iterasyonda sıfırla
for epoch in range(10):
    optimizer.zero_grad()
    loss = model(x)
    loss.backward()
    optimizer.step()
```

### 4️⃣ Custom Autograd
```python
class MyFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # Kaydet!
        return input * 2
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2  # Türev
```

---

## ⚠️ Yaygın Hatalar

### ❌ HATA 1: View'da Non-Contiguous
```python
x = torch.randn(3, 4)
y = x.t()
z = y.view(12)  # RuntimeError!

# ÇÖZÜM
z = y.contiguous().view(12)
```

### ❌ HATA 2: GPU Synchronize Unutmak
```python
# YANLIŞ
start = time.time()
y = x_gpu @ x_gpu
time_taken = time.time() - start  # Yanlış!

# DOĞRU
start = time.time()
y = x_gpu @ x_gpu
torch.cuda.synchronize()  # Bekle!
time_taken = time.time() - start
```

### ❌ HATA 3: In-place İşlem Gradient Graph'ı Bozar
```python
x = torch.tensor([1.0], requires_grad=True)
y = x**2

y.add_(1.0)  # RuntimeError!
y.backward()

# ÇÖZÜM
y = y.add(1.0)  # Yeni tensor döndür
```

---

## 📖 Ek Kaynaklar

### Resmi Dokümantasyon
- [PyTorch Tensor Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)

### Akademik Makaleler
- "Automatic Differentiation in PyTorch" (Paszke et al., 2017)
- "Anatomy of High-Performance Matrix Multiplication" (Goto & Van De Geijn)
- "CUDA Programming Guide" (NVIDIA)

### Video Kaynaklar
- Andrej Karpathy - "Neural Networks: Zero to Hero"
- PyTorch Internals - Edward Yang
- CUDA Programming - NVIDIA Developer

---

## 🎯 Sonraki Adım

**Faz 2: Neural Network Fundamentals**

Artık tensor mekanikleri ve autograd'ı biliyorsunuz. Sırada:
- Linear Regression (sıfırdan, matematik ile)
- nn.Module mimarisi
- Activation functions (ReLU, GELU, Swish)
- Loss functions (CrossEntropy, MSE)
- Optimizers (SGD, Adam, AdamW)

```bash
cd ../Faz_2_Neural_Networks
```

---

## ✅ Başarı Kriterleri

Bu fazı tamamladıysanız:

- [ ] Bir tensor'un stride'ını hesaplayabiliyorsunuz
- [ ] View vs reshape farkını açıklayabiliyorsunuz
- [ ] CPU-GPU transfer darboğazını tespit edebiliyorsunuz
- [ ] Gradient flow'u debug edebiliyorsunuz
- [ ] Kendi activation function'ınızı yazabiliyorsunuz
- [ ] Numerical gradient checking yapabiliyorsunuz

**Hepsini işaretlediyseniz, Faz 2'ye geçebilirsiniz!** 🚀

---

**Son Güncelleme:** 18 Ocak 2026  
**Durum:** ✅ Tamamlandı (7/7 ders)
