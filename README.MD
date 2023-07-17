# Cómo instalar las librerías

# 1.- Instalar pipenv.
En la terminal se escribe
pip install pipenv

# 2.- Tener instalado python 3.10 o superior.

# 3.- En la terminal ir al path de la carpeta que contiene todos estos archivos (Automatizacion_levitacion-main).
Escribir:
pipenv install.

# 4.- Asegurarse de tener ejecutado el entorno virtual.
Si no esta ejecutado escribir en la terminal estando en el path de esta carpeta
pipenv shell

# 5.- Ejecutar un archivo usando:
pipenv run python <archivo.py>
El archivo donde se pone en práctica todo es "game.py"


# ...
Ya ejecutado "main.py"
en la terminal se escoge un modelo entrenado.
Originalmente la carpeta archivos_generados cuenta con 5 modelo entrenados.
Por esta razón se debe ingresar un número del 1 al 5.
El juego ha sido iniciado.
Ahora se puede cambiar la posición de referencia con las flechas de arriba y abajo desde el juego.
También se puede ingresar directamente una posición de referencia en la terminal.
Se podrá apreciar como el gráfico se mueve junto a la pelota, ya que, el gráfico
es una representación de la posición de la pelota en función del tiempo.
La red neuronal entrega una presión cada vez que cambiamos la posición de referencia,
esta nueva presión la podemos ver en el cuadrado azul debajo del tubo.