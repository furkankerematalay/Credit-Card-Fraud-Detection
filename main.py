# Veritabanı İşlemleri
import pandas as pd
import numpy as np
import sqlite3

# Veri Hazırlığı
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler, Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline

# Derin Öğrenme (TensorFlow & Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns
import random as rn

sns.set(style='whitegrid', context='notebook')


df = pd.read_csv('creditcard.csv')

conn = sqlite3.connect('fraud_detection.db')
df.to_sql('transactions', conn, if_exists='replace', index=False)

test_query = "SELECT Time, Amount, Class FROM transactions LIMIT 5"
test_df = pd.read_sql(test_query, conn)
print(test_df)

random_seed = 42
training_sample = 200000
validate_size = 0.2

np.random.seed(random_seed)
rn.seed(random_seed)
tf.random.set_seed(random_seed)

df.columns = map(str.lower,df.columns)
df.rename(columns={'class': 'label'},inplace=True)

print(df.head())

scaler = RobustScaler()
df[['amount','time']] = scaler.fit_transform(df[['amount','time']])

fraud = df[df.label == 1]
clean = df[df.label == 0].sample(n=2000, random_state=42)

mini_df = pd.concat([fraud,clean])
X = mini_df.drop('label',axis=1)
Y = mini_df['label']

tsne_model = TSNE(n_components=2,random_state=42)
x_2d = tsne_model.fit_transform(X)

plt.figure(figsize=(10,8))

sns.scatterplot(
    x=x_2d[:,0],
    y=x_2d[:,1],
    hue =Y,
    palette='coolwarm',
    alpha=0.7,
    linewidth=0
)

plt.title('t-SNE')
plt.xlabel(' X')
plt.ylabel(' Y')
plt.legend(title='label', labels=['Normal (0)', 'Fraud (1)'])

print(plt.show())

normal_df = df[df.label == 0]
fraud_df = df[df.label == 1]

train_df,test_normal_df = train_test_split(normal_df,test_size=0.2 ,random_state=42)

test_df =pd.concat([test_normal_df,fraud_df])

x_train = train_df.drop(['label', 'time'], axis=1)
x_test = test_df.drop(['label', 'time'],axis=1)
y_test = test_df['label']

x_train, x_validate = train_test_split(x_train, test_size=0.2, random_state=42)

x_train = x_train.values
x_validate = x_validate.values
x_test = x_test.values
y_test = y_test.values

pipeline = Pipeline([('normalizer', Normalizer()),
                    ('scaler', MinMaxScaler())])

pipeline.fit(x_train);

x_train_transformed = pipeline.transform(x_train)
x_validate_transformed = pipeline.transform(x_validate)
x_test_transformed = pipeline.transform(x_test)


input_dim = x_train_transformed.shape[1]
autoencoder = Sequential()

autoencoder.add(Dense(units=16, input_dim=input_dim, activation='elu'))
autoencoder.add(Dense(units=8, activation='elu'))
autoencoder.add(Dense(units=4, activation='elu'))
autoencoder.add(Dense(units=2, activation='elu'))

autoencoder.add(Dense(units=4, activation='elu'))
autoencoder.add(Dense(units=8, activation='elu'))
autoencoder.add(Dense(units=input_dim, activation='elu'))

autoencoder.compile(optimizer="adam", loss="mse", metrics=["acc"])
# İşlemciye "Eğer 10 tur boyunca val_loss düşmezse eğitimi acil durdur" emri
early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True # RAM'deki en kusursuz matrisi geri yükle
)

# İşlemciye "Her rekor kırdığında (val_loss düştüğünde) SSD'ye kaydet" emri
save_model = ModelCheckpoint(
    filepath='autoencoder_best_weights.h5', # Hard diskteki dosya adı
    save_best_only=True,
    monitor='val_loss',
    verbose=0,
    mode='min'
)
cb = [early_stop, save_model]

history = autoencoder.fit(
    x_train_transformed, x_train_transformed,
    shuffle=True, # RAM'deki adresleri her tur karıştır (Ezberi boz)
    epochs=100,   # 100 tur yetkisi verdik ama bekçi muhtemelen erken kesecek
    batch_size=256,
    callbacks=cb, # Bekçileri eğitim döngüsünün (for loop) içine enjekte ettik
    validation_data=(x_validate_transformed, x_validate_transformed)


)
reconstructions = autoencoder.predict(x_test_transformed)
mse_skorlari = np.mean(np.power(x_test_transformed - reconstructions, 2), axis=1)


# revize MAD ---
def mad_score(points):
    """ Modifiye Edilmiş Z-Skoru (Modified Z-Score) Hesaplayıcı """
    # İşlemci tüm hataların tam ortasını (Medyan) bulur
    m = np.median(points)

    # Her bir hatanın bu merkeze olan mutlak uzaklığını hesaplar (Vektörel)
    ad = np.abs(points - m)

    # Bu uzaklıkların da medyanını alarak 'Normal Esneme Payını' bulur
    mad = np.median(ad)

    # Matematiksel standartlaştırma (0.6745 sabiti ile)
    return 0.6745 * ad / mad


# İşlemci, test setindeki tüm mse_skorlari'ni bu fonksiyona sokar
# Çıktı olarak her bir işlem için bir 'Sapma Skoru' (Z-Score) üretir
z_scores = mad_score(mse_skorlari)

#  MAD için kabul edilen evrensel anomali sınırı 3'tür.
THRESHOLD_MAD = 2.5


anomalies = z_scores > THRESHOLD_MAD

predictions = anomalies.astype(int)

from sklearn.metrics import confusion_matrix, classification_report

print("--- Hata Matrisi (Confusion Matrix) ---")
print(confusion_matrix(y_test, predictions))

print("\n--- Sınıflandırma Raporu (Classification Report) ---")
print(classification_report(y_test, predictions))