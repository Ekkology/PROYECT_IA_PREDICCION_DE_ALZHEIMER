import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

# Leer el archivo CSV
data = pd.read_csv('/home/ekkology/repositorios/PROYECT_IA_PREDICCION_DE_ALZHEIMER/Notebooks/datos.csv')

# Obtener las características y las etiquetas del DataFrame
features = data[['contrast', 'energy', 'intensity']]
labels = data['label']

# Convertir etiquetas de texto a valores numéricos
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

# Crear el modelo de red neuronal
model = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
accuracy = model.score(X_test, y_test)
print("Exactitud del modelo:", accuracy)
