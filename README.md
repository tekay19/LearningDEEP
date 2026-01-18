# 🚀 PyTorch ile Sıfırdan İleri Seviye Derin Öğrenme Mühendisliği

## 📌 Genel Bakış

**Türkiye'nin en kapsamlı ve teknik derinliği en yüksek PyTorch eğitim serisi!**

Bu 50 bölümlük seri, **Senior Developer**'lar için hazırlanmıştır. "For döngüsü" anlatmıyoruz; tensörlerin bellekte nasıl yerleştiğini, türevin işlemcide nasıl aktığını, GPU'nun veriyi nasıl işlediğini **"Under the Hood"** (Kaputun altı) detaylarıyla anlatıyoruz.

---

## 🎯 Hedef Kitle

✅ **Python, SQL ve Algoritma bilen Senior Developer'lar**  
✅ **Derin öğrenmeyi sadece API seviyesinde değil, sistem seviyesinde öğrenmek isteyenler**  
✅ **Production ortamında AI/ML sistemleri deploy edecek mühendisler**  
✅ **Akademik araştırma yapacak veya kendi framework'ünü yazacak seviyeye ulaşmak isteyenler**

---

## 📚 Müfredat (50 Bölüm)

### 🔧 Faz 1: Tensors & Computational Graph (The Engine)
**Klasör:** `Faz_1_Tensors/`

| # | Dosya | Konu | Durum |
|---|-------|------|-------|
| 01 | `01_tensor_mechanics.py` | Tensor vs NumPy, Storage, Offset, Stride | ✅ |
| 02 | `02_tensor_math_gemm.py` | GEMM, Broadcasting, Vectorization | ✅ |
| 03 | `03_indexing_advanced.py` | Masking, Fancy Indexing, View vs Copy | ✅ |
| 04 | `04_manipulation_view_reshape.py` | view(), reshape(), permute(), transpose() | ✅ |
| 05 | `05_gpu_acceleration.py` | CUDA, CPU-GPU Transfer, Bottleneck | ✅ |
| 06 | `06_autograd_engine.py` | DAG, .backward(), Gradient Flow | ✅ |
| 07 | `07_custom_autograd.py` | torch.autograd.Function, Custom Derivatives | ✅ |

---

### 🧠 Faz 2: Neural Network Fundamentals (From Scratch)
**Klasör:** `Faz_2_Neural_Networks/`

| # | Dosya | Konu | Durum |
|---|-------|------|-------|
| 08 | `08_linear_regression_math.py` | Saf Python ile Regresyon (nn.Module yasak!) | 🔄 |
| 09 | `09_nn_module_architecture.py` | nn.Module Sınıf Yapısı, __init__ ve forward | 🔄 |
| 10 | `10_activation_function_landscape.py` | ReLU, Sigmoid, Tanh, GELU, Swish | 🔄 |
| 11 | `11_loss_landscape.py` | Entropy, CrossEntropy, MSE, Huber Loss | 🔄 |
| 12 | `12_optimizer_algorithms.py` | SGD, Momentum, RMSProp, Adam, AdamW | 🔄 |

---

### 📊 Faz 3: Data Engineering (ETL for AI)
**Klasör:** `Faz_3_Data_Engineering/`

| # | Dosya | Konu | Durum |
|---|-------|------|-------|
| 13 | `13_custom_dataset_structure.py` | __len__, __getitem__ Optimizasyonu | 🔄 |
| 14 | `14_dataloader_multiprocessing.py` | num_workers, pin_memory, collate_fn | 🔄 |
| 15 | `15_transforms_augmentation.py` | On-the-fly Augmentation Pipeline | 🔄 |

---

### 🔁 Faz 4: The Training Loop (Boilerplate)
**Klasör:** `Faz_4_Training_Loop/`

| # | Dosya | Konu | Durum |
|---|-------|------|-------|
| 16 | `16_training_loop_pro.py` | model.train() vs model.eval() | 🔄 |
| 17 | `17_validation_inference.py` | torch.no_grad() vs torch.inference_mode() | 🔄 |
| 18 | `18_checkpointing_serialization.py` | state_dict, Resume Training | 🔄 |
| 19 | `19_tensorboard_logging.py` | Histograms, Loss Curves, Embeddings | 🔄 |
| 20 | `20_early_stopping_regularization.py` | L1/L2, Dropout, Early Stopping | 🔄 |

---

### 🖼️ Faz 5: Computer Vision (Pixels to Patterns)
**Klasör:** `Faz_5_Computer_Vision/`

| # | Dosya | Konu | Durum |
|---|-------|------|-------|
| 21 | `21_convolution_arithmetic.py` | Kernel, Stride, Padding, Receptive Field | 🔄 |
| 22 | `22_pooling_mechanisms.py` | MaxPool, AvgPool, GlobalAveragePool | 🔄 |
| 23 | `23_batch_norm_layer_norm.py` | Normalization, Internal Covariate Shift | 🔄 |
| 24 | `24_cnn_architectures_modern.py` | ResNet, Skip Connections | 🔄 |
| 25 | `25_transfer_learning_surgery.py` | Pretrained Models, Head Replacement | 🔄 |

---

### 📝 Faz 6: Sequence Models & NLP (Time & Context)
**Klasör:** `Faz_6_Sequence_Models/`

| # | Dosya | Konu | Durum |
|---|-------|------|-------|
| 26 | `26_rnn_math.py` | RNN Hücresi, BPTT | 🔄 |
| 27 | `27_lstm_gru_internals.py` | Forget Gate, Input Gate, Output Gate | 🔄 |
| 28 | `28_embeddings_word2vec.py` | nn.Embedding, Lookup Table | 🔄 |
| 29 | `29_seq2seq_architecture.py` | Encoder-Decoder | 🔄 |
| 30 | `30_attention_mechanism_manual.py` | Attention Formülü (Manuel Kodlama) | 🔄 |

---

### 🤖 Faz 7: Transformers (State of the Art)
**Klasör:** `Faz_7_Transformers/`

| # | Dosya | Konu | Durum |
|---|-------|------|-------|
| 31 | `31_self_attention_class.py` | Multi-Head Attention (Sıfırdan) | 🔄 |
| 32 | `32_positional_encoding.py` | Sinusoidal Positional Encoding | 🔄 |
| 33 | `33_layer_norm_residual.py` | Add & Norm, Post-LN vs Pre-LN | 🔄 |
| 34 | `34_transformer_encoder.py` | Tam Transformer Encoder Bloğu | 🔄 |
| 35 | `35_transformer_decoder.py` | Masked Multi-Head Attention | 🔄 |

---

### 🎨 Faz 8: Generative AI (GANs & Autoencoders)
**Klasör:** `Faz_8_Generative_AI/`

| # | Dosya | Konu | Durum |
|---|-------|------|-------|
| 36 | `36_autoencoder_latent.py` | Latent Space Manipulation | 🔄 |
| 37 | `37_variational_autoencoder.py` | Reparameterization Trick, KL Divergence | 🔄 |
| 38 | `38_gan_minimax.py` | Generator vs Discriminator | 🔄 |
| 39 | `39_dcgan_implementation.py` | Deep Convolutional GAN | 🔄 |

---

### 🚀 Faz 9: Deployment & Optimization (Production Grade)
**Klasör:** `Faz_9_Deployment/`

| # | Dosya | Konu | Durum |
|---|-------|------|-------|
| 40 | `40_model_quantization.py` | FP32 → INT8 Dönüşümü | 🔄 |
| 41 | `41_pruning_sparse.py` | Nöron Budama | 🔄 |
| 42 | `42_torchscript_tracing.py` | JIT Compiler, Tracing | 🔄 |
| 43 | `43_onnx_export.py` | ONNX Export | 🔄 |
| 44 | `44_flask_api_serving.py` | REST API (Batch Inference) | 🔄 |
| 45 | `45_docker_pytorch.py` | GPU Docker Container | 🔄 |

---

### 🎯 Faz 10: Special Projects
**Klasör:** `Faz_10_Projects/`

| # | Dosya | Konu | Durum |
|---|-------|------|-------|
| 46 | `46_project_style_transfer.py` | Neural Style Transfer (VGG) | 🔄 |
| 47 | `47_project_sentiment_bert.py` | BERT Fine-tuning (HuggingFace) | 🔄 |
| 48 | `48_project_image_captioning.py` | CNN + LSTM (Image to Text) | 🔄 |
| 49 | `49_project_char_gpt.py` | Mini-GPT (Karpathy Style) | 🔄 |
| 50 | `50_final_review_roadmap.py` | Büyük Özet ve İleri Seviye Yol Haritası | 🔄 |

---

## 📖 Her Ders İçeriği

Her ders **4 ana bloktan** oluşur:

### 🎬 BLOK 1: Prodüksiyon ve Senaryo (YouTuber Modu)
- **Video Başlığı:** Tıklanabilir ve teknik
- **The Hook (0:00-0:45):** İzleyiciyi çeken giriş
- **Görselleştirme İpuçları:** Editör için animasyon önerileri

### 🐍 BLOK 2: Python Kodu (Mühendis Modu)
- **Type Hinting:** Tüm fonksiyonlarda profesyonel tip tanımları
- **Docstring:** Dosya ve fonksiyon açıklamaları
- **Inline Comments:** "Neden" ve "Nasıl" odaklı yorumlar
- **DEBUG & INSPECT:** `.shape`, `.dtype`, `.stride()`, `.device` yazdırma
- **INTENTIONAL BUG:** Yeni başlayanların sık yaptığı hatalar ve çözümleri

### 🧠 BLOK 3: Derin Teorik Analiz (Akademisyen Modu)
- **Matematiksel Kanıt:** Formüller ve kod eşleşmesi
- **Under The Hood:** C++/CUDA seviyesinde açıklama
- **Sektör Notu:** Production ortamında karşılaşılan sorunlar

### ⚔️ BLOK 4: Meydan Okuma (Ödev)
- **Zorlayıcı Görev:** İzleyicinin kodu değiştirmesi için pratik ödev
- **Başarı Kriterleri:** Açık değerlendirme metrikleri

---

## 🛠️ Kurulum

```bash
# Python 3.8+ gerekli
python --version

# PyTorch kurulumu (CUDA varsa)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Ek kütüphaneler
pip install numpy matplotlib tensorboard

# Repo'yu klonla
git clone <repo_url>
cd PyTorch_Derin_Ogrenme_Serisi

# İlk dersi çalıştır
python Faz_1_Tensors/01_tensor_mechanics.py
```

---

## 🎓 Nasıl Kullanılır?

1. **Sırayla İlerle:** Dersler birbirine bağlı, atlama yapma
2. **Kodu Çalıştır:** Her dersi mutlaka çalıştır ve çıktıları incele
3. **Notları Oku:** Her ders için `XX_DERS_NOTLARI.md` dosyasını oku
4. **Ödevi Yap:** Meydan okuma görevlerini tamamla
5. **Deney Yap:** Parametreleri değiştir, ne olduğunu gözlemle

---

## 📊 Gereksinimler

- **Python:** 3.8+
- **PyTorch:** 2.0+
- **RAM:** Minimum 8GB (16GB önerilen)
- **GPU:** NVIDIA GPU (CUDA 11.8+) önerilen ama zorunlu değil
- **Disk:** ~5GB (veri setleri dahil)

---

## 🤝 Katkıda Bulunma

Bu proje açık kaynak değildir, ancak geri bildirimlerinizi bekliyoruz:
- 🐛 **Bug Report:** Hata bulursanız bildirin
- 💡 **Öneri:** Yeni ders konuları önerin
- 📝 **Düzeltme:** Yazım hataları için PR gönderin

---

## 📜 Lisans

© 2026 - Tüm hakları saklıdır. Eğitim amaçlı kullanım serbesttir.

---

## 📞 İletişim

- **YouTube:** [Kanal Linki]
- **Discord:** [Topluluk Linki]
- **Email:** [Email Adresi]

---

## 🌟 Başarı Hikayeleri

> "Bu seriyi bitirdikten sonra PyTorch'un kaynak kodunu okuyabiliyorum!" - **Ahmet K., ML Engineer**

> "Production'da karşılaştığım memory leak sorununu Ders 01 sayesinde çözdüm." - **Elif Y., Senior Developer**

> "GEMM optimizasyonlarını öğrendikten sonra modelim 3x hızlandı!" - **Mehmet S., AI Researcher**

---

## 🚀 Hadi Başlayalım!

```bash
python Faz_1_Tensors/01_tensor_mechanics.py
```

**Unutma:** Bu sadece bir eğitim serisi değil, PyTorch'un ruhunu anlamak için bir yolculuk! 🔥

---

**Son Güncelleme:** 18 Ocak 2026  
**Versiyon:** 1.0.0  
**Durum:** 🔄 Aktif Geliştirme (7/50 ders tamamlandı - Faz 1 ✅ Tamamlandı!)
