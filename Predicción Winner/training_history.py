import pandas as pd
import matplotlib.pyplot as plt

# Leer el historial de entrenamiento
history_df = pd.read_csv('training_history_rnn.csv')

# Graficar la pérdida
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_df['loss'], label='Pérdida entrenamiento')
plt.plot(history_df['val_loss'], label='Pérdida validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Evolución de la Pérdida')
plt.legend()

# Graficar la precisión
plt.subplot(1, 2, 2)
plt.plot(history_df['accuracy'], label='Precisión entrenamiento')
plt.plot(history_df['val_accuracy'], label='Precisión validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.title('Evolución de la Precisión')
plt.legend()

plt.tight_layout()
plt.show()