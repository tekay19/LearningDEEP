# PyTorch Derin Öğrenme Notları

PyTorch öğrenirken tuttuğum notlar ve kod örnekleri. Temel tensor işlemlerinden production deployment'a kadar 50 ders.

## Durum

**Tamamlanan:** 12/50 ders (%24)  
**Son güncelleme:** 18 Ocak 2026

## Tamamlanan Bölümler

### ✅ Faz 1: Tensors & Autograd (7 ders)

Tensor'ların bellekte nasıl çalıştığını, GPU transfer optimizasyonunu ve autograd mekanizmasını öğrendim.

1. **Tensor Mechanics** - Storage, stride, contiguous memory
2. **GEMM & Broadcasting** - Matris çarpımı optimizasyonu, BLAS
3. **Advanced Indexing** - Masking, fancy indexing, view vs copy
4. **View & Reshape** - Bellek optimizasyonu teknikleri
5. **GPU Acceleration** - CUDA, pinned memory, streams
6. **Autograd Engine** - DAG yapısı, gradient hesaplama
7. **Custom Autograd** - Kendi türev fonksiyonunu yazma

**Öğrendiklerim:**
- Stride mekanizması sayesinde transpose işlemi zero-copy
- CPU-GPU transfer darboğazı nasıl önlenir
- Manuel gradient vs autograd karşılaştırması
- Custom activation function nasıl yazılır

### ✅ Faz 2: Neural Networks (5 ders)

nn.Module'den optimizer'lara kadar neural network temellerini işledim.

8. **Linear Regression** - Sıfırdan, matematik ile (nn.Module yasak!)
9. **nn.Module Architecture** - Parameter yönetimi, hooks, state_dict
10. **Activation Functions** - ReLU, GELU, Swish, vanishing gradient
11. **Loss Functions** - MSE, CrossEntropy, Focal Loss
12. **Optimizers** - SGD, Momentum, Adam, AdamW

**Öğrendiklerim:**
- Manuel gradient descent vs autograd
- nn.Module'ün içinde ne oluyor
- Hangi activation ne zaman kullanılır
- Adam vs AdamW farkı (weight decay)

## Planlanan Bölümler

### 🔄 Faz 3: Data Engineering (3 ders)

Dataset ve DataLoader optimizasyonları.

13. Custom Dataset - `__len__`, `__getitem__` optimizasyonu
14. DataLoader - num_workers, pin_memory, collate_fn
15. Transforms - On-the-fly augmentation pipeline

### 🔄 Faz 4: Training Loop (5 ders)

Production-grade training loop yazma.

16. Training Loop - model.train() vs model.eval()
17. Validation - torch.no_grad() vs inference_mode()
18. Checkpointing - state_dict, resume training
19. Logging - TensorBoard, loss curves
20. Regularization - L1/L2, dropout, early stopping

### 🔄 Faz 5: Computer Vision (5 ders)

CNN'ler ve modern mimariler.

21. Convolution - Kernel, stride, padding, receptive field
22. Pooling - MaxPool, AvgPool, GlobalAveragePool
23. Normalization - BatchNorm, LayerNorm
24. CNN Architectures - ResNet, skip connections
25. Transfer Learning - Pretrained models, fine-tuning

### 🔄 Faz 6: NLP & Sequences (5 ders)

RNN'ler ve attention mekanizması.

26. RNN - Recurrent networks, BPTT
27. LSTM/GRU - Forget gate, input gate, output gate
28. Embeddings - nn.Embedding, Word2Vec
29. Seq2Seq - Encoder-decoder architecture
30. Attention - Attention mechanism (manuel)

### 🔄 Faz 7: Transformers (5 ders)

Modern NLP'nin temeli.

31. Self-Attention - Multi-head attention (sıfırdan)
32. Positional Encoding - Sinusoidal encoding
33. Layer Norm - Add & Norm blocks
34. Transformer Encoder - Tam encoder bloğu
35. Transformer Decoder - Masked attention, causal masking

### 🔄 Faz 8: Generative AI (4 ders)

VAE ve GAN'lar.

36. Autoencoder - Latent space manipulation
37. VAE - Reparameterization trick, KL divergence
38. GAN - Generator vs discriminator
39. DCGAN - Deep convolutional GAN

### 🔄 Faz 9: Deployment (6 ders)

Production'a alma.

40. Quantization - FP32 → INT8 dönüşümü
41. Pruning - Model budama
42. TorchScript - JIT compiler, tracing
43. ONNX Export - Cross-platform deployment
44. API Serving - Flask/FastAPI ile serving
45. Docker - GPU container hazırlama

### 🔄 Faz 10: Projects (5 ders)

Gerçek projeler.

46. Style Transfer - Neural style transfer (VGG)
47. BERT Fine-tuning - HuggingFace ile sentiment analysis
48. Image Captioning - CNN + LSTM
49. Mini-GPT - Character-level language model
50. Final Review - Özet ve ileri seviye yol haritası

## Kullanım

```bash
git clone https://github.com/tekay19/LearningDEEP.git
cd LearningDEEP

# Faz 1
cd Faz_1_Tensors
python 01_tensor_mechanics.py

# Faz 2
cd ../Faz_2_Neural_Networks
python 08_linear_regression_math.py
```

Her ders çalıştırılabilir Python kodu. Bazı derslerin yanında ders notları da var.

## Gereksinimler

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib (görselleştirme için)

GPU opsiyonel, tüm kodlar CPU'da da çalışır.

## İlerleme

```
Faz 1: ████████████████████ 100% (7/7)
Faz 2: ████████████████████ 100% (5/5)
Faz 3: ░░░░░░░░░░░░░░░░░░░░   0% (0/3)
Faz 4: ░░░░░░░░░░░░░░░░░░░░   0% (0/5)
Faz 5: ░░░░░░░░░░░░░░░░░░░░   0% (0/5)
Faz 6: ░░░░░░░░░░░░░░░░░░░░   0% (0/5)
Faz 7: ░░░░░░░░░░░░░░░░░░░░   0% (0/5)
Faz 8: ░░░░░░░░░░░░░░░░░░░░   0% (0/4)
Faz 9: ░░░░░░░░░░░░░░░░░░░░   0% (0/6)
Faz 10: ░░░░░░░░░░░░░░░░░░░░  0% (0/5)

Toplam: ████░░░░░░░░░░░░░░░░ 24% (12/50)
```

## Notlar

- Her ders type-hinted ve documented
- Production-ready kod örnekleri
- Manuel implementasyonlar (nn.Module kullanmadan)
- GPU optimizasyonları dahil

## Lisans

Eğitim amaçlı kullanım serbest.
