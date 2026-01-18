# PyTorch ile Derin Öğrenme

PyTorch'u sıfırdan öğrenmek için hazırladığım kapsamlı bir eğitim serisi. Temel tensor işlemlerinden production deployment'a kadar her şey var.

## Neden bu repo?

Türkçe PyTorch kaynakları genelde yüzeysel kalıyor. Ben sadece API'leri göstermekle kalmayıp, altında ne olduğunu da anlatmak istedim. Bellekte stride nasıl çalışır, GPU transfer neden yavaş, autograd nasıl gradient hesaplar gibi konulara giriyorum.

## İçerik

Toplam 50 ders planladım, şu an 7 tanesi hazır.

### Faz 1: Tensors & Computational Graph ✅

1. **Tensor Mechanics** - Storage, stride, contiguous memory
2. **GEMM & Broadcasting** - Matris çarpımı optimizasyonu
3. **Advanced Indexing** - Masking, fancy indexing
4. **View & Reshape** - Bellek optimizasyonu
5. **GPU Acceleration** - CUDA, pinned memory, streams
6. **Autograd Engine** - Gradient hesaplama mekanizması
7. **Custom Autograd** - Kendi türev fonksiyonunu yazma

### Gelecek Fazlar

- Faz 2: Neural Networks (Linear regression, nn.Module, activations, loss, optimizers)
- Faz 3: Data Engineering (Dataset, DataLoader, augmentation)
- Faz 4: Training Loop (Checkpointing, logging, early stopping)
- Faz 5: Computer Vision (CNN, ResNet, transfer learning)
- Faz 6: NLP & Sequences (RNN, LSTM, attention)
- Faz 7: Transformers (Self-attention, encoder-decoder)
- Faz 8: Generative AI (VAE, GAN)
- Faz 9: Deployment (Quantization, ONNX, TorchScript)
- Faz 10: Projeler (Style transfer, BERT fine-tuning, image captioning)

## Nasıl kullanılır?

```bash
git clone https://github.com/tekay19/LearningDEEP.git
cd LearningDEEP/Faz_1_Tensors
python 01_tensor_mechanics.py
```

Her ders çalıştırılabilir Python kodu. Bazı dersler için ders notları da var.

## Gereksinimler

- Python 3.8+
- PyTorch 2.0+
- NumPy

GPU opsiyonel. Kodlar CPU'da da çalışır.

## Katkıda bulunma

Hata bulursanız veya öneri varsa issue açabilirsiniz.

## Lisans

Eğitim amaçlı kullanım serbest.
