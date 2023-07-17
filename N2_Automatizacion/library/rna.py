import hernan_olmi_simulacion_pelota_levitacion as Simulacion
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from typing import Union

MODEL_PATH = './archivos_generados/ventilador_red_neuronal5.keras'
SAMPLE_DATA_PATH= './archivos_generados/sample_data.csv'

class RNA:
    """Crea una red neuronal."""
    def __init__(self, cantidad_entradas:int = None, tasa_aprendizaje:float = None, epocas:int = None, datos_entrenamiento:str = None):
        """
        cantidad_entradas: Entradas de la capa de entrada.
        tasa_aprendizaje: Magnitud en la que se ajustan los pesos en cada propagación hacia atrás.
        epocas: Cuantas veces se verifica el error de los pesos actuales y se hace el ajuste.
        datos_entrenamiento: path de archivo .csv con datos de entrenamiento y testeo.
        """
        self.cantidad_entradas: int = cantidad_entradas
        self.entradas: np.ndarray = np.array([])
        self.salidas: np.ndarray = np.array([])
        self.entradas_test: np.ndarray = np.array([])
        self.salidas_test: np.ndarray = np.array([])
        self.tasa_aprendizaje: float = tasa_aprendizaje
        self.epocas: int = epocas
        self.datos_entrenamiento: str = datos_entrenamiento
        self.modelo: tf.keras.Model = None
        self.nombre_modelo: str = None

    def crear_modelo_nuevo(self, entradas: list[str], targets: list[str]) -> None:
        """
        Entrenar la red neuronal.
        entradas: Especificación de que representan las entradas.
        targets: Especificación de que representan las salidas.
        """
        df = pd.read_csv(self.datos_entrenamiento)
        self.entradas = df.loc[:, entradas]
        self.salidas = df.loc[:, targets]

        x_train = [] # Datos de entrenamiento
        x_test = [] # Datos de testeo

        # Dejando 80% para entrenamiento
        val_intervalo = round(len(self.entradas) * 0.8)
        x_train, x_test, y_train, y_test = self.entradas[:val_intervalo], self.entradas[val_intervalo:], self.salidas[:val_intervalo], self.salidas[val_intervalo:]

        x_train = np.array(x_train, dtype=float)
        y_train = np.array(y_train, dtype=float)
        x_test = np.array(x_test, dtype=float)
        y_test = np.array(y_test, dtype=float)

        self.entradas = x_train
        self.salidas = y_train
        self.entradas_test = x_test
        self.salidas_test = y_test

        # Crear modelo. Esta es una red neuronal de 3 capas:
        # 1.- Entrada: Es una neurona con una entrada que es una posicion.
        # 2.- Oculta: Capa con 200 neuronas, cada una recibe una entrada.
        # 3.- Salida: Es una neurona que recibe 200 entradas.
        modelo = tf.keras.models.Sequential()
        modelo.add(tf.keras.layers.Dense(self.cantidad_entradas, input_shape=(self.cantidad_entradas, )))  # Capa entrada
        modelo.add(tf.keras.layers.Dense(200, activation='relu'))  # Capa oculta
        modelo.add(tf.keras.layers.Dense(1, activation='linear'))  # Capa salida

        # Optimizadores
        modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= self.tasa_aprendizaje, epsilon=0.1), loss='mean_squared_error', metrics=['accuracy'])

        # Entrenar
        historial = modelo.fit(x_train, y_train, epochs=self.epocas, verbose=False)

        # Ver relacion entre épocas y loss para saber si es necesario menos épocas para el modelo
        plt.plot(historial.history["loss"])
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.show()

        # Mostrar rendimiento de modelo entrenado
        os.system('cls')
        loss, acc = modelo.evaluate(x_test, y_test, verbose=2)
        print("Loss:", loss)
        print("Accuracy:", acc)

        # Guardar modelo entrenado si es que no existe
        if os.path.exists(MODEL_PATH):
            eleccion = input("¿Quieres sobreescribir el modelo ya existente? [1: SI] [2: NO]: ")
            eleccion = int(eleccion)
            if eleccion == 1:
                modelo.save(MODEL_PATH)
            else:
                print("Utilizando modelo antiguo")
        else:
            print("Guardando modelo")
            modelo.save(MODEL_PATH)

        self.modelo = modelo

    def cargar_modelo_existente(self, x: list[str], y: list[str], datos_entrenamiento: str) -> Union[tf.keras.Model, None]:
        """
        Retorna un modelo para el archivo .keras
        x: Valores de entrada a la red.
        y: Valores objetivo de la red.
        datos_entrenamiento: Path al archivo .csv que contiene los datos de entrenamiento y testeo.
        """
        if os.path.exists(MODEL_PATH):
            self.modelo = tf.keras.models.load_model(MODEL_PATH)
            self.nombre_modelo = MODEL_PATH
            self.datos_entrenamiento = datos_entrenamiento
            df = pd.read_csv(self.datos_entrenamiento, encoding='latin-1')
            self.entradas = df.loc[:, x]
            self.salidas = df.loc[:, y]
            return self.modelo
        else:
            print("No se encontró un modelo existente.")

    def predecir(self, entrada_prediccion: list[list[float]]) -> float:
        """
        Predice la presión necesaria para posicion/es de referencia.
        entrada_prediccion: Posición o posiciones a las cuales se les quiere predecir la presión necesaria.
        """
        if self.modelo is None:
            print("No se ha creado o cargado un modelo.")
            return

        entrada_conversion = np.array(entrada_prediccion)

        if len(entrada_conversion.shape) != 2 or entrada_conversion.shape[1] != 1:
            print(f"La entrada de predicción debe tener forma (n, {self.cantidad_entradas}).")
            return

        return self.modelo.predict(entrada_prediccion)[0][0]
    
##################################################################################################
# FUNCIONES GLOBALES
##################################################################################################
def predecir_presion_menu() -> None:
    """Menu para interactuar con la red neuronal mandandole posiciones y que nos diga la presión predecida."""
    entrada_correcta = False
    posicion = 0
    while entrada_correcta != True:
        try:
            posicion = input("Ingrese una posición preferentemente entre 0 [m] y 1 [m]: ")
            posicion_ = float(posicion)
            entrada_correcta = True
        except:
            print("Ingrese una entrada que se pueda convertir a decimal")
    entrada_prediccion = [[posicion_]]
    presion = modelo.predecir(entrada_prediccion)

    # Se instancia una planta
    planta = Simulacion.Planta (
        pelota = Simulacion.Pelota(),
        tubo = Simulacion.Tubo(),
        ventilador = Simulacion.Ventilador()
    )
    corutina = planta()
    delta_tiempo: float = 0
    _ , _ , delta_tiempo, _, _ = corutina.send(None)
    ejex = []
    ejey = []

    tiempo_elegido = input("Elija un tiempo de simulación en segundos: ")
    tiempo_elegido = int(float(tiempo_elegido) / delta_tiempo)
    for _ in range(tiempo_elegido):
        posicion, tiempo, _, _, _ = corutina.send(presion)
        ejex.append(tiempo)
        ejey.append(posicion)
    print(f"Posicion RF: {posicion_}, Posición real: {posicion}, Presión enviada: {presion} [kg/m^3], Error: {posicion_ - posicion}")
    print("Tiempo:", tiempo, "[s]")
    planta.graficar(ejex, ejey)

def resumen_modelo_menu() -> None:
    """Resumen de las características de la red neuronal."""
    rna.summary()

def ver_rendimiento_modelo_menu() -> None:
    """Evaluación de que tan buena para predecir es nuestra red neuronal."""
    df = pd.read_csv(SAMPLE_DATA_PATH, encoding='latin-1')
    entradas = df.loc[:, ['posicion']]
    salidas = df.loc[:, ['presion']]    

    posiciones = entradas
    presiones = salidas

    x_train = [] # Datos de entrenamiento
    x_test = [] # Datos de testeo

    val_intervalo = round(len(posiciones) * 0.8)
    x_train, x_test, y_train, y_test = posiciones[:val_intervalo], posiciones[val_intervalo:], presiones[:val_intervalo], presiones[val_intervalo:]

    x_train = np.array(x_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    x_test = np.array(x_test, dtype=float)
    y_test = np.array(y_test, dtype=float)

    loss, acc = rna.evaluate(x_test, y_test, verbose=2)
    print("Loss:", loss)
    print("Accuracy:", acc)

##################################################################################################
# MAIN
##################################################################################################

if __name__ == '__main__':
    print(os.path.dirname(__file__))
    if os.path.exists(SAMPLE_DATA_PATH):
        while True:
            eleccion = input(f"[(1) Utilizar modelo existente]\n[(2) Crear nuevo modelo]\n[(3) Salir]\n: ")
            eleccion = int(eleccion)
            os.system('cls')
            match eleccion:
                # Utilizar modelo existente  
                case 1:
                    modelo = RNA(datos_entrenamiento= MODEL_PATH)
                    rna = modelo.cargar_modelo_existente(x=['posicion'], y=['presion'], datos_entrenamiento=SAMPLE_DATA_PATH)
                    os.system('cls')
                    while True:
                        opcion = input(f"[(1) Predecir una presión para llegar a una posición]\n[(2) Ver resumen de modelo <{modelo.nombre_modelo}>]\n[(3) Ver rendimiento de modelo]\n[(4) Ir al menú]\n: ")
                        opcion = int(opcion)
                        os.system('cls')
                        match opcion:
                            case 1:
                                predecir_presion_menu()
                            case 2:
                                resumen_modelo_menu()
                            case 3:
                                ver_rendimiento_modelo_menu()
                            case 4:
                                break
                            case other:
                                os.system('cls')
                                print("Elija una opción válida")

                # Crear nuevo modelo
                case 2:
                    while True:
                        verificacion = False
                        try:
                            tasa = input("Ingrese un valor para la tasa de aprendizaje [RECOMENDACION: que vaya entre (0 - 1)]\n: ")
                            tasa = float(tasa)
                            verificacion = True
                            epocas = input("Ingrese la cantidad de épocas para entrenar el modelo\n :")
                            epocas = int(epocas)
                            break
                        except:
                            os.system('cls')
                            if verificacion == False:
                                print("Ingrese una tasa de aprendizaje que se pueda convertir en FLOAT")
                            else:
                                print("Ingrese una cantidad de épocas que se pueda convertir en INTEGER")

                    modelo = RNA(cantidad_entradas=1, tasa_aprendizaje=tasa, epocas=epocas, datos_entrenamiento=SAMPLE_DATA_PATH)
                    modelo.crear_modelo_nuevo(['posicion'], ['presion'])
                    os.system('cls')
                # Salir
                case 3:
                    break

                case other:
                    print("Elija una opción válida")
    else:
        print("¡¡¡Primero debes generar datos en 'consiguiendo_datos.py'!!!")