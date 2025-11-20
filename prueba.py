import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model("best_digit_classifier.h5")
print("Modelo cargado correctamente")

def preprocess_digit(roi):
    """
    Mejora el preprocesamiento del dígito para que coincida con cómo fue entrenado el modelo
    """
    # Asegurarse de que el ROI no esté vacío
    if roi.size == 0:
        return None
    
    # Redimensionar a 28x28 manteniendo la relación de aspecto
    h, w = roi.shape
    if h == 0 or w == 0:
        return None
    
    # Crear una imagen cuadrada para mantener las proporciones
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    
    # Centrar el dígito en la imagen cuadrada
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = roi
    
    # Redimensionar a 28x28
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalizar como durante el entrenamiento
    normalized = resized.astype('float32') / 255.0
    
    return normalized

def improve_binarization(image):
    """
    Mejora la binarización usando diferentes métodos
    """
    # Método 1: Threshold adaptativo (mejor para condiciones variables de luz)
    binary_adapt = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
    
    # Método 2: Otsu's threshold
    _, binary_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Probar ambos y elegir el que tenga mejor relación señal/ruido
    adapt_white_pixels = np.sum(binary_adapt > 0)
    otsu_white_pixels = np.sum(binary_otsu > 0)
    
    # Preferir el que tenga una cantidad razonable de píxeles blancos
    total_pixels = image.shape[0] * image.shape[1]
    adapt_ratio = adapt_white_pixels / total_pixels
    otsu_ratio = otsu_white_pixels / total_pixels
    
    # Si la proporción está entre 0.1 y 0.5, probablemente sea un dígito bien formado
    if 0.1 <= adapt_ratio <= 0.5:
        return binary_adapt
    elif 0.1 <= otsu_ratio <= 0.5:
        return binary_otsu
    else:
        # Si ninguno es ideal, usar adaptativo por defecto
        return binary_adapt

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Configurar resolución para mejor calidad
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Sistema de reconocimiento iniciado")
print("Presiona 'q' para salir")
print("Presiona 'd' para modo depuración")

debug_mode = False
stable_predictions = {}  # Para estabilizar predicciones entre frames

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: No se pudo capturar el frame")
            break
        
        # Voltear horizontalmente para efecto espejo
        frame = cv2.flip(frame, 1)
        
        # Crear una copia para mostrar resultados
        display_frame = frame.copy()
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplicar desenfoque - ajustar tamaño según necesidad
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Mejorar binarización
        binary = improve_binarization(blurred)
        
        # Operaciones morfológicas para limpiar la imagen
        kernel = np.ones((2, 2), np.uint8)
        # Primero opening para eliminar ruido pequeño
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        # Luego closing para unir partes rotas del dígito
        kernel_closing = np.ones((3, 3), np.uint8)
        dilated = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_closing)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Ordenar contornos por área (de mayor a menor)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        digits_detected = 0
        
        for i, contour in enumerate(contours):
            if digits_detected >= 3:  # Limitar a 3 dígitos máximo por frame
                break
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filtrar por tamaño - criterios más estrictos
            min_area = 500  # Área mínima
            max_area = 50000  # Área máxima
            aspect_ratio = w / h if h > 0 else 0
            
            area = w * h
            if (area < min_area or area > max_area or 
                w < 20 or h < 30 or 
                aspect_ratio < 0.2 or aspect_ratio > 2.0):
                continue
            
            # Extraer ROI de la imagen binaria original
            roi = binary[y:y+h, x:x+w]
            
            if roi.size == 0:
                continue
                
            # Preprocesar para el modelo
            processed_roi = preprocess_digit(roi)
            
            if processed_roi is None:
                continue
            
            # Preparar para predicción
            roi_input = processed_roi.reshape(1, 28, 28, 1)
            
            # Realizar predicción
            prediction = model.predict(roi_input, verbose=0)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Filtrar por confianza
            if confidence < 0.6:  # Solo mostrar predicciones confiables
                continue
            
            # Estabilizar predicciones (evitar cambios bruscos entre frames)
            contour_id = f"{x//10}_{y//10}"  # ID aproximado basado en posición
            if contour_id in stable_predictions:
                old_digit, old_confidence, count = stable_predictions[contour_id]
                if digit == old_digit:
                    count += 1
                    confidence = (old_confidence * count + confidence) / (count + 1)
                else:
                    count = 1
                stable_predictions[contour_id] = (digit, confidence, min(count, 5))
            else:
                stable_predictions[contour_id] = (digit, confidence, 1)
            
            # Usar la predicción estabilizada
            stable_digit, stable_confidence, _ = stable_predictions[contour_id]
            
            # Dibujar resultados
            color = (0, 255, 0) if stable_confidence > 0.8 else (0, 255, 255)
            thickness = 3 if stable_confidence > 0.8 else 2
            
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, thickness)
            
            # Texto con dígito y confianza
            text = f"{stable_digit} ({stable_confidence:.2f})"
            cv2.putText(display_frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            digits_detected += 1
            
            # Mostrar ROI procesado en modo depuración
            if debug_mode and digits_detected <= 2:
                roi_display = (processed_roi * 255).astype(np.uint8)
                roi_display = cv2.resize(roi_display, (100, 100))
                roi_bgr = cv2.cvtColor(roi_display, cv2.COLOR_GRAY2BGR)
                
                # Posicionar en esquinas diferentes
                pos_x = 10 + (digits_detected - 1) * 110
                display_frame[10:110, pos_x:pos_x+100] = roi_bgr
                cv2.putText(display_frame, f"ROI {digits_detected}", 
                           (pos_x, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mostrar información general
        cv2.putText(display_frame, f"Digitos: {digits_detected}", 
                   (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostrar frame principal
        cv2.imshow("Reconocimiento de Digitos Mejorado", display_frame)
        
        # Mostrar vistas de depuración si está activado
        if debug_mode:
            # Mostrar imagen binaria
            binary_display = cv2.resize(binary, (320, 240))
            cv2.imshow("1. Binaria", binary_display)
            
            # Mostrar imagen dilatada/limpiada
            dilated_display = cv2.resize(dilated, (320, 240))
            cv2.imshow("2. Limpiada", dilated_display)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Modo depuracion: {'ACTIVADO' if debug_mode else 'DESACTIVADO'}")
            if not debug_mode:
                # Cerrar ventanas de depuración
                cv2.destroyWindow("1. Binaria")
                cv2.destroyWindow("2. Limpiada")

except KeyboardInterrupt:
    print("Interrupcion por teclado")
except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Sistema terminado")