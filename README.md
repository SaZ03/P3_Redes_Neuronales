# P3_Redes_Neuronales
# Proyecto: Reconocimiento de Dígitos con Redes Neuronales Convolucionales

## Descripción del Proyecto
Este proyecto implementa un sistema completo de reconocimiento óptico de caracteres (OCR) para dígitos escritos a mano utilizando redes neuronales convolucionales. Incluye el entrenamiento de múltiples arquitecturas de CNN, evaluación comparativa y una aplicación en tiempo real que detecta y clasifica dígitos mediante la cámara web.

## Características Principales
- **6 arquitecturas CNN diferentes** con variaciones en regularización y profundidad
- **Análisis comparativo detallado** de rendimiento y overfitting
- **Sistema OCR en tiempo real** con detección y clasificación de múltiples dígitos
- **Preprocesamiento avanzado** de imágenes para mejorar la detección
- **Evaluación de técnicas de regularización** (BatchNorm, Dropout)

## Archivos del Proyecto

### Código y Análisis
* **Notebook principal con entrenamiento y análisis** → [P3_Redes_Neuronales_Sergio_Zamora.ipynb](./P3_Redes_Neuronales_Sergio_Zamora.ipynb)
* **Versión HTML del notebook** → [P3_Redes_Neuronales_Sergio_Zamora.html](./P3_Redes_Neuronales_Sergio_Zamora.html)
* **Aplicación OCR en tiempo real** → [EnVivoOCR.py](./EnVivoOCR.py)

### Modelos y Documentación
* **Modelo entrenado final** → [best_digit_classifier.h5](./best_digit_classifier.h5)
* **Presentación del proyecto** → [Presentacion.pdf](./Presentacion.pdf)

### Recursos Externos
* **Video demostración del sistema** → [Ver Video](https://drive.google.com/file/d/178Bdlcwd-usDaoOcpLQJRD7au5JKle1C/view?usp=sharing)
* Profesor, porfavor ponga el video en x2, no logré llegar a los 30 segundos y tampoco encontré una forma rápida de editar el video para que esté en x2, una disculpa
* **Dataset de dígitos para entrenamiento** → [Acceder al Dataset](https://drive.google.com/drive/folders/1UOh0TLqPrPh1odW5VYact01mTGMM98_h?usp=drive_link)

## Metodología Implementada

### 1. Entrenamiento de Modelos
- **6 arquitecturas CNN**: Simple, BatchNorm, Dropout, Profunda, Kernels Mixtos y Combinada
- **Ajuste experimental** de hiperparámetros y técnicas de regularización
- **Early Stopping** para prevenir overfitting
- **Validación cruzada** con 30% de los datos

### 2. Procesamiento de Imágenes
- **Binarización adaptativa** para diferentes condiciones de iluminación
- **Operaciones morfológicas** para limpiar y unir componentes
- **Detección de contornos** y extracción de ROI
- **Preprocesamiento consistente** con el entrenamiento

### 3. Sistema OCR en Tiempo Real
- **Detección múltiple** de dígitos por frame
- **Estabilización de predicciones** entre frames consecutivos
- **Visualización con bounding boxes** y niveles de confianza
- **Modo depuración** para análisis del procesamiento

## Resultados Obtenidos
El modelo **Modelo 4 - Profundo** demostró el mejor rendimiento con 80.45% de exactitud en validación. Las técnicas de **Dropout** mostraron mayor efectividad en controlar el overfitting comparado con BatchNorm. El sistema final logró un 80.28% de exactitud en el conjunto de test, manteniendo un buen balance entre rendimiento y generalización.

## Conclusiones
El proyecto demostró la efectividad de las CNN para reconocimiento de dígitos y la importancia de las técnicas de regularización para mejorar la generalización. El sistema implementado es capaz de detectar y clasificar múltiples dígitos en tiempo real con buen rendimiento, siendo extensible para reconocer otros caracteres y patrones.
