# Alzheimer Evreleri Sınıflandırma (ResNet50 & Transfer Learning)

Bu proje, bir "Bilgi Sistemleri" dersi kapsamında geliştirilmiştir. Amacı, beyin MR (Manyetik Rezonans) görüntülerinden faydalanarak Alzheimer hastalığının 4 farklı evresini (Hafif, Çok Hafif, Orta, Demented Olmayan) sınıflandıran bir derin öğrenme modeli (CNN) oluşturmaktır.

## 📋 Proje Özeti

  * **Problem:** Çok sınıflı görüntü sınıflandırma (4 sınıf).
  * **Veri Seti:** Kaggle'daki [Alzheimer's Classification Dataset](https://www.kaggle.com/datasets/kanaadlimaye/alzheimers-classification-dataset).
  * **Model:** ResNet50 (ImageNet üzerinde ön-eğitimli).
  * **Teknik:** Transfer Learning (Transfer Öğrenme) ve Fine-Tuning (İnce Ayar).
  * **Sonuç:** Model, daha önce hiç görmediği test verisi üzerinde **%86.56**'lık bir doğruluk (accuracy) elde etmiştir.

-----

## 🧠 Metodoloji

Proje, `ResNet50` mimarisi üzerine kurulu iki aşamalı bir eğitim stratejisi izlemiştir:

### 1\. Aşama: Transfer Learning (Özellik Çıkarımı)

  * ImageNet veri setiyle eğitilmiş hazır `ResNet50` modelinin "gövdesi" (evrişimsel katmanları) donduruldu (`trainable = False`).
  * Modelin sonuna, kendi 4 sınıfımızı (MD, MoD, ND, VMD) sınıflandıracak özel bir "kafa" katmanı eklendi (`GlobalAveragePooling2D`, `Dense(1024)`, `Dropout(0.5)` ve `Dense(4, 'softmax')`).
  * Model, 10 epoch boyunca sadece bu yeni "kafa" katmanlarını eğitmek üzere çalıştırıldı.
  * **Sonuç:** 10. epoch sonunda **\~%63** doğrulama (validation) başarısı elde edildi.

### 2\. Aşama: Fine-Tuning (İnce Ayar)

  * İlk aşamada elde edilen %63'lük başarıyı artırmak için, dondurulan `ResNet50` gövdesi "çözüldü" (`trainable = True`).
  * Modelin ImageNet'ten öğrendiği değerli bilgileri bozmamak için, öğrenme oranı (learning rate) çok düşük bir değere (`1e-5` yani `0.00001`) çekildi.
  * Model, bu düşük öğrenme oranıyla 10 epoch daha (toplam 20 epoch) eğitildi.
  * **Sonuç:** Bu "ince ayar" hamlesi, modelin beyin MR görüntülerindeki ince nüansları da öğrenmesini sağladı ve doğrulama başarısını **\~%85** seviyesine çıkardı.

-----

## 📈 Sonuçlar ve Performans

Modelin 20 epoch'luk eğitim süreci boyunca gösterdiği gelişim aşağıdaki grafiklerde özetlenmiştir. 10. epoch'ta (gri kesikli çizgi) başlayan "İnce Ayar" hamlesinin, modelin başarımını (turuncu çizgi) nasıl keskin bir şekilde artırdığı açıkça görülmektedir.

*(Bu `README.md` dosyasıyla aynı klasöre `model_performans_grafigi.png` dosyasını da yüklediğinden emin ol.)*
<br>
\<img src="model\_performans\_grafigi.png" alt="Model Performans Grafiği" width="800"/\>

### Nihai Test Sonucu

Model, eğitim ve doğrulama aşamalarında hiç görmediği `test` veri seti üzerinde son kez değerlendirilmiş ve aşağıdaki nihai sonucu almıştır:

| Metrik | Skor |
| :--- | :--- |
| **Test Kaybı (Loss)** | `0.3469` |
| **Test Başarısı (Accuracy)** | **`%86.56`** |

-----

## 🚀 Proje Dosyaları ve Kullanımı

Bu repo, modeli eğitmek ve test etmek için gerekli 3 ana dosyayı içerir:

### 1\. Gereksinimler

Projeyi çalıştırmak için gerekli kütüphaneler:

```bash
pip install tensorflow pandas matplotlib
```

### 2\. Dosya Yapısı

Projenin çalışması için klasör yapısı şu şekilde olmalıdır:

```
.
├── Alzheimers_Detection_dataset/   <-- (Kaggle'dan indirilen veri seti)
│   ├── CSV_datafiles/
│   ├── test/
│   ├── train/
│   └── valid/
├── proje.py                        <-- (Modeli sıfırdan eğiten ana script)
├── tahmin_et.py                    <-- (Eğitilmiş modeli test etmek için script)
├── alzheimer_resnet50_model.h5     <-- (Eğitilmiş modelin kayıtlı 'beyni')
└── README.md                       <-- (Bu dosya)
```

**⚠️ ÖNEMLİ NOT:** `alzheimer_resnet50_model.h5` dosyası (300+ MB) GitHub'ın 100MB'lık dosya limitinden büyüktür. Bu dosyayı repoya yüklemek için [Git LFS (Large File Storage)](https://git-lfs.github.com/) kullanmanız veya `.gitignore` dosyasına ekleyip, modeli (örn: Google Drive) üzerinden harici olarak paylaşmanız gerekir.

### 3\. Modelin Sıfırdan Eğitilmesi

Modeli baştan sona (20 epoch) eğitmek için:

```bash
python proje.py
```

Bu script, eğitim tamamlandığında `alzheimer_resnet50_model_final.h5` (veya `proje.py` içinde ne ad verdiyseniz o) adıyla modeli ve `model_performans_grafigi.png` dosyasını oluşturacaktır.

### 4\. Eğitilmiş Model ile Tahmin Yapma

Elinizdeki `.h5` modelini kullanarak `test` klasöründen rastgele bir MR görüntüsünü sınıflandırmak için:

```bash
python tahmin_et.py
```

**Örnek Çıktı:**

```
Model 'alzheimer_resnet50_model.h5' başarıyla yüklendi.
Rastgele test görüntüsü seçildi: MD_24_jpg.rf.0a1b2c3d...
--- TAHMİN SONUCU ---
Görüntü: MD_24_jpg.rf.0a1b2c3d...
Modelin Tahmini: MD (Mild Demented)
Emin Olma Skoru: 91.82 %
```
