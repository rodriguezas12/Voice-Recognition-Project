import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Escalador global
scaler = StandardScaler()

# Función para extraer características mejoradas
def extract_features(audio_file):
    audio, sample_rate = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    combined = np.vstack((mfcc, delta_mfcc))
    combined = np.mean(combined, axis=1)
    return combined

# Cargar datos
def load_data(directory):
    features = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            audio_file = os.path.join(directory, filename)
            feature = extract_features(audio_file)
            if 'english' in filename.lower():
                label = 0
            elif 'arabic' in filename.lower():
                label = 1
            elif 'mandarin' in filename.lower():
                label = 2
            else:
                continue
            features.append(feature)
            labels.append(label)
    return np.array(features), np.array(labels)

# Directorio y carga
directory = r"C:\Users\Sergio\Desktop\ASDASDASD\Tareitas u.u\Decimo semestre\Intro to IA\Stage 3\nnnnn\nnnnn\prueba\audio"
features, labels = load_data(directory)
features = scaler.fit_transform(features)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Entrenar modelo con Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar con validación cruzada
scores = cross_val_score(model, features, labels, cv=5)
print(f"Cross-validated accuracy: {np.mean(scores) * 100:.2f}%")

# Predicciones y métricas
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["English", "Arabic", "Mandarin"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Función para clasificar un nuevo archivo
def classify_new_audio(audio_file):
    if not os.path.exists(audio_file):
        print(f"Error: El archivo {audio_file} no existe.")
        return
    feature = extract_features(audio_file)
    feature = scaler.transform([feature])  # Normalizar el nuevo archivo
    probabilities = model.predict_proba(feature)[0]  # Obtener probabilidades
    labels = ["inglés", "árabe", "mandarín"]
    
    for i, label in enumerate(labels):
        print(f"Este audio tiene {probabilities[i] * 100:.2f}% de probabilidad de ser {label}.")
    
    # Obtener la clase con mayor probabilidad
    max_index = np.argmax(probabilities)
    print(f"Este audio tiene acento {labels[max_index]}.\n")


# Pruebas
print("\n=== Pruebas con archivos específicos ===")
test_files = [
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\1.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\2.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\3.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\4.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\5.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\6.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\7.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\8.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\9.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\10.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\11.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\12.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\13.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\14.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\15.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\16.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\17.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\18.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\19.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\20.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\21.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\22.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\23.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\24.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\25.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\26.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\27.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\28.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\29.wav", 
    r"C:\Users\kolet\Downloads\Intro to AI\prueba\audio\30.wav"
]

for test_file in test_files:
    print(f"Probando archivo: {test_file}")
    classify_new_audio(test_file)
