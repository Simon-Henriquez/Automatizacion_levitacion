"""Autor del código: Simón Henríquez Lind
Simulación de la planta creada por Hernan Olmí y encargado de letic Oscar, para el ramo de automatización.
Esta aplicación intenta replicar el comportamiento de una pelota levitando dentro de un tubo, debido a presiones de entrada.

Para usar esta aplicación se debe crear un objeto <Planta()>, luego crear la instancia corutina llamando a la instancia de planta como función <objeto_planta()>
y listo!, ya se puede comenzar a enviar presiones, y se retornarán los datos actuales de la pelota.
Para enviar presiones se haría algo como esto corutina.send(500).

AL final del código se encuentra un ejemplo en la función <app> de como usar este programa."""
import matplotlib.pyplot as plt

from typing import Generator

class Pelota:
    def __init__(self, radio: float = 0.1, densidad: int = 20):
        self.radio = radio
        self.densidad = densidad
        self.posicion = 0
        self.aceleracion = 0
        self.velocidad = 0

class Tubo:
    def __init__(self, largo: int = 1):
        self.largo = largo

class Ventilador:
    def __init__(self, presion: int = 0):
        self.presion = presion

class Planta:
    def __init__(
            self,
            pelota: Pelota,
            tubo: Tubo,
            ventilador: Ventilador,
            delta_tiempo: float = 0.005,
            gravedad: float = 9.8,
            densidad_fluido: float = 20
        ):

        ### Atributos
        self.tiempo_actual = 0
        self.DELTA_TIEMPO = delta_tiempo
        self.GRAVEDAD = gravedad
        self.DENSIDAD_FLUIDO = densidad_fluido
        ### Objetos, composición
        self.pelota = pelota
        self.tubo = tubo
        self.ventilador = ventilador

        self.DENOMINADOR_FORMULA_ACELERACION = self.pelota.densidad * (2/3) * self.pelota.radio

    def calculando_aceleracion(self, posicion_anterior: float, presion: float) -> float:
        """Calcula la aceleración de la pelota debido a un cambio en la presión del ventilador."""
        return (
            (
                (presion - (self.DENSIDAD_FLUIDO * self.GRAVEDAD * posicion_anterior))
                /
                self.DENOMINADOR_FORMULA_ACELERACION
            )
            - self.GRAVEDAD
        )

    def calcular_posicion(self, posicion_anterior: float, velocidad_anterior: float) -> float:
        """Calcula la posición nueva de la pelota debido a un cambio en su aceleración."""
        return (
            posicion_anterior
            # +
            # (velocidad_anterior * self.delta_tiempo) # Esto esta comentado ya que en el codigo original v[iteracion] siempre era cero.
            +
            (0.5 * self.pelota.aceleracion * pow(self.DELTA_TIEMPO,2))
        )

    def __call__(self, imprimir: bool = False) -> Generator[tuple[int], int, None]:
        """Coroutine, ejecucion de la simulacion, asincronica al ingreso de presiones, por lo tanto recibe una presión.
        Si queremos imprimir cada resultado imprimir=true.
        Esta corutina retorna una tupla con los siguientes 5 datos de la planta
        <(posicion, tiempo_actual, delta_tiempo, aceleracion, velocidad)>."""

        if imprimir:
            g_impresion = self.impresion()
            g_impresion.send(None)
        while True:
            # Recibiendo presion de ventilador para el tiempo actual
            presion = yield (self.pelota.posicion, self.tiempo_actual, self.DELTA_TIEMPO, self.pelota.aceleracion, self.pelota.velocidad)
            self.ventilador.presion = presion

            # Datos a usar como valores anteriores "k-1"
            posicion_anterior = self.pelota.posicion
            velocidad_anterior = self.pelota.velocidad

            # Calculando nueva aceleración y nueva posición
            self.pelota.aceleracion = self.calculando_aceleracion(posicion_anterior, presion)
            self.pelota.posicion = self.calcular_posicion(posicion_anterior, velocidad_anterior)

            # Revisando si se respetan los limites del tubo, si no, pelota en reposo
            if self.pelota.posicion > self.tubo.largo:
                self.pelota.posicion = self.tubo.largo
                self.pelota.aceleracion = 0
                self.pelota.velocidad = 0
            elif self.pelota.posicion < 0:
                self.pelota.posicion = 0
                self.pelota.aceleracion = 0
                self.pelota.velocidad = 0

            # Imprime el resultado
            if imprimir:
                g_impresion.send({"Posicion_actual": self.pelota.posicion, "Tiempo_actual": self.tiempo_actual, "Delta_tiempo": self.DELTA_TIEMPO, "Aceleracion": self.pelota.aceleracion, "Velocidad": self.pelota.velocidad, "Presion": presion})

            # Actualiza el tiempo
            self.tiempo_actual += self.DELTA_TIEMPO

            # Cambiando la velocidad de la pelota para el proximo incremento de tiempo
            self.pelota.velocidad += self.pelota.aceleracion * self.DELTA_TIEMPO

    def impresion(self) -> Generator[None, dict[str, int], None]:
        """Imprime los resultados que manda la corutina."""
        while True:
            diccionario: dict = yield
            print("###############################################")
            for key, value in diccionario.items():
                print(f"{key}: {value}")

    def graficar(self, x: list[int], y: list[int]):
        """Graficar un grafico de linea."""
        plt.plot(x, y, color='blue' )
        plt.xlabel('Tiempo')
        plt.ylabel('Posición')
        plt.show()

# App de ejemplo
def app():
    ##########################
    # Iniciando la simulacion
    ##########################
    # Creando la planta
    planta = Planta(pelota=Pelota(), tubo=Tubo(), ventilador=Ventilador())

    # Creando la corutina de la simulación, para esto llamamos a la planta como si fuera una función
    corutina = planta(imprimir=False) # imprimir=True -> Imprime los resultados cada vez que despertamos la corutina
    corutina.send(None) # Priming, le mandamos None para que se inicie la simulación
    ##########################
    # Simulacion ya inició
    ##########################


    ##########################
    # Usando la simulación
    ##########################
    ejex = [] # ejex del grafico, "tiempo"
    ejey = [] # ejey del grafico, "posición"

    for presion in range(400): # recorriendo nuestras presiones
        posicion, tiempo, delta_tiempo, aceleracion, velocidad = corutina.send(presion) # Enviando presión a la planta, esta retornará la tupla <(posicion, tiempo_actual, delta_tiempo, aceleracion, velocidad)>
        #Datos para graficar
        ejex.append(tiempo)
        ejey.append(posicion)

    planta.graficar(ejex, ejey) # Graficando datos
    # Se puede seguir usando la simulación si se desea y esta continuará donde quedo...
    # Para esto solo hay que volver a enviar presiones con <corutina.send(presion)> ...
    # ...

if __name__ == "__main__":
    print("\n--- APP ---\n")
    print("Cargando...")
    app()