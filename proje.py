import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
import os

print(f"TensorFlow Sürümü: {tf.__version__}")

# Veri setinin bulunduğu ana klasör
base_dir = 'Alzheimers_Detection_dataset'

# Veri setinin yolları
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

csv_train_path = os.path.join(base_dir, 'CSV_datafiles', '_train_classes.csv')
csv_valid_path = os.path.join(base_dir, 'CSV_datafiles', '_valid_classes.csv')
csv_test_path = os.path.join(base_dir, 'CSV_datafiles', '_test_classes.csv')

# Model parametreleri
IMAGE_SIZE = 224
BATCH_SIZE = 32
class_labels = ['MD', 'MoD', 'ND', 'VMD'] 

print("Veri yolları ve parametreler tanımlandı.")

# CSV dosyalarını Pandas DataFrame'e yükle
try:
    train_df = pd.read_csv(csv_train_path)
    valid_df = pd.read_csv(csv_valid_path)
    test_df = pd.read_csv(csv_test_path)
    print("CSV dosyaları başarıyla okundu.")
except FileNotFoundError as e:
    print(f"Hata: CSV dosyası bulunamadı. {e}")
    print("Lütfen 'Alzheimers_Detection_dataset' klasörünün doğru yerde olduğundan emin olun.")
    exit() # Hata varsa script durur

# Eğitim Jeneratörü (Veri Artırma + Normalizasyon)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Doğrulama ve Test Jeneratörü (Sadece Normalizasyon)
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Jeneratörleri DataFrame'lere bağla
print("Veri jeneratörleri kuruluyor...")

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col='filename',
    y_col=class_labels,
    class_mode='raw',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory=validation_dir,
    x_col='filename',
    y_col=class_labels,
    class_mode='raw',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col='filename',
    y_col=class_labels,
    class_mode='raw',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Veri jeneratörleri başarıyla kuruldu.")

# Temel ResNet50 Modelini Yükle (Ön-eğitimli, son katmansız)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# Temel Modelin Katmanlarını Dondur
base_model.trainable = False

# Özel Sınıflandırma Katmanlarını Ekle
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x) # 4 sınıf için çıkış katmanı

# Yeni Modeli Birleştir
model = Model(inputs=base_model.input, outputs=predictions)

# Modeli Derle (Aşama 1: Transfer Learning)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Model başarıyla kuruldu ve derlendi.")
model.summary()

# --- Eğitim Aşaması 1 (Transfer Learning) ---
print("\n--- Eğitim Aşaması 1 (Transfer Learning) Başlıyor ---")
history = model.fit(
    train_generator,
    epochs=10, 
    validation_data=validation_generator
)
print("Eğitim Aşaması 1 tamamlandı.")

# --- Eğitim Aşaması 2 (Fine-Tuning) ---
print("\n--- Eğitim Aşaması 2 (Fine-Tuning) Başlıyor ---")

# Temel Modelin Katmanlarını Çöz
base_model.trainable = True

# Modeli Çok Düşük Bir Öğrenme Oranıyla Yeniden Derle
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), # 1e-5
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model 'ince ayar' için çözüldü ve yeniden derlendi.")
model.summary() # Eğitilebilir parametre sayısını kontrol et

# İnce Ayar Eğitimini Başlat
history_fine_tune = model.fit(
    train_generator,
    epochs=20,
    initial_epoch=history.epoch[-1] + 1, # Kaldığı epoch'tan devam et
    validation_data=validation_generator
)
print("Eğitim Aşaması 2 (Fine-Tuning) tamamlandı.")

# Eğitilen Modeli Kaydet
model_filename = 'alzheimer_resnet50_model_final.h5'
model.save(model_filename)
print(f"Model başarıyla '{model_filename}' adıyla kaydedildi.")

# Modeli Test Verisi ile Değerlendir
print("\nModelin test verisi üzerindeki performansı ölçülüyor...")
test_loss, test_accuracy = model.evaluate(test_generator)

print("\n--- NİHAİ TEST SONUÇLARI ---")
print(f"Test Kaybı (Test Loss):     {test_loss:.4f}")
print(f"Test Başarısı (Test Accuracy): {test_accuracy * 100:.2f} %")

# --- Sonuçları Görselleştirme ---
print("\nPerformans grafikleri oluşturuluyor...")

# İki eğitim aşamasının geçmiş verilerini birleştir
acc = history.history['accuracy'] + history_fine_tune.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine_tune.history['val_accuracy']

loss = history.history['loss'] + history_fine_tune.history['loss']
val_loss = history.history['val_loss'] + history_fine_tune.history['val_loss']

epochs_range = range(1, len(acc) + 1)

# Doğruluk (Accuracy) Grafiği
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Eğitim Başarısı (Training Accuracy)')
plt.plot(epochs_range, val_acc, label='Doğrulama Başarısı (Validation Accuracy)')
plt.axvline(x=10, color='grey', linestyle='--', label='İnce Ayar Başlangıcı') # 10. epoch'ta ince ayar başladı
plt.legend(loc='lower right')
plt.title('Eğitim ve Doğrulama Başarısı (Accuracy)')
plt.xlabel('Epoch Sayısı')
plt.ylabel('Doğruluk')

# Kayıp (Loss) Grafiği
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Eğitim Kaybı (Training Loss)')
plt.plot(epochs_range, val_loss, label='Doğrulama Kaybı (Validation Loss)')
plt.axvline(x=10, color='grey', linestyle='--', label='İnce Ayar Başlangıcı') # 10. epoch'ta ince ayar başladı
plt.legend(loc='upper right')
plt.title('Eğitim ve Doğrulama Kaybı (Loss)')
plt.xlabel('Epoch Sayısı')
plt.ylabel('Kayıp')

# Grafiği kaydet
grafik_dosya_adi = 'model_performans_grafigi.png'
plt.savefig(grafik_dosya_adi)
print(f"Grafikler '{grafik_dosya_adi}' adıyla kaydedildi.")
# plt.show() 

print("\n--- Proje Kodu Tamamlandı ---")