# PyTorch Derin Öğrenme Notları

PyTorch öğrenirken tuttuğum notlar ve kod örnekleri. Temel tensor işlemlerinden başlayıp production deployment'a kadar gidiyor.

## Neden?

Türkçe PyTorch kaynaklarının çoğu sadece API kullanımını gösteriyor. Ben biraz daha derine inip altında ne olduğunu anlamaya çalıştım. Mesela stride nasıl çalışır, GPU transfer neden yavaş olabilir, autograd gradient'leri nasıl hesaplar gibi.

## İçerik

Şu an 7 ders hazır, toplamda 50 ders olacak.

### Hazır Olanlar

**Faz 1: Tensors & Autograd**
- Tensor mechanics (storage, stride, contiguous memory)
- GEMM & broadcasting (matris çarpımı optimizasyonu)
- Advanced indexing (masking, fancy indexing)
- View & reshape (bellek optimizasyonu)
- GPU acceleration (CUDA, pinned memory, streams)
- Autograd engine (gradient hesaplama)
- Custom autograd (kendi türev fonksiyonunu yazma)

### Planlananlar

- Neural networks (linear regression, nn.Module, activations, loss, optimizers)
- Data engineering (Dataset, DataLoader, augmentation)
- Training loop (checkpointing, logging, early stopping)
- Computer vision (CNN, ResNet, transfer learning)
- NLP (RNN, LSTM, attention)
- Transformers (self-attention, encoder-decoder)
- Generative models (VAE, GAN)
- Deployment (quantization, ONNX, TorchScript)
- Projeler (style transfer, BERT fine-tuning, image captioning)

## Kullanım

```bash
git clone https://github.com/tekay19/LearningDEEP.git
cd LearningDEEP/Faz_1_Tensors
python 01_tensor_mechanics.py
```

Her ders çalıştırılabilir Python kodu. Bazılarının yanında ders notları da var.

## Gereksinimler

- Python 3.8+
- PyTorch 2.0+
- NumPy

GPU opsiyonel, kodlar CPU'da da çalışır.

## Lisans

Eğitim amaçlı kullanım serbest.
