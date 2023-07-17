import hernan_olmi_simulacion_pelota_levitacion as Simulacion
import numpy as np
import csv

# Creando la planta
planta = Simulacion.Planta (
    pelota = Simulacion.Pelota(),
    tubo = Simulacion.Tubo(),
    ventilador = Simulacion.Ventilador()
)

# Iniciando la simulación
corutina = planta()
corutina.send(None)
######################################

data = [("posicion", "presion")]
ejex = []
ejey = []

MIN_P = 20 # Mínima presion que puede dar el ventilador para elevar la pelota
MAX_P = 215 # Desde esta presión la pelota se eleva a 1 metro a lo largo de 10 segundos
presiones = np.random.uniform(MIN_P, MAX_P, MAX_P*2)

print("Cargando...")

for presion in presiones:
    for _ in range(4000): # Se le da 10 segundos para que la pelota se estabilice
        posicion, tiempo, _, _, _ = corutina.send(presion)
    ejex.append(tiempo)
    ejey.append(posicion)
    # Guardando la posición de la pelota para una presión luego de 10 segundos
    data.append((posicion, presion))

# Guardando los valores de la posición de la pelota para cada presión
with open('archivos_generados/sample_data.csv', 'w', newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)

planta.graficar(ejex, ejey)