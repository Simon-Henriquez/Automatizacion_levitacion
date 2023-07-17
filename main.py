import pygame, os, signal, threading
import tensorflow as tf
import library.hernan_olmi_simulacion_pelota_levitacion as Simulacion
class PosicionFueraDeRangoError(Exception):
    pass

class Grafico:
    def __init__(self):
        self.datos: list[tuple[int,int]] = []
        self.tiempo_inicial = 0
        self.tiempo_actual = 0
        self.ejey = TOP
        self.ejex = pc_rect.right+100
        self.cero = goku_rect.top-POS*2
        self.divisiones = 10
        self.espaciado = 500 / self.divisiones-1 

    def update(self, y: int, x: float):
        pos = map_range(y, 0, 100, self.cero, self.ejey+self.espaciado/2)
        self.datos.append((x, pos))
        self.tiempo_actual = x

    def mostrar(self):
        pygame.draw.lines(display_surface, 'white', False, ((self.ejex, self.ejey),(self.ejex, self.cero), (self.ejex+self.cero, self.cero)), width=2)
        for line in range(self.divisiones+1): # eje y
            y_cord = map_range(line, 10, 0, self.ejey, self.cero)
            pygame.draw.line(display_surface, 'white', (self.ejex, y_cord), (self.ejex-10, y_cord), width=1)
            text_surf = ball_position_font.render(str(10*line), True, 'white')
            display_surface.blit(
                text_surf,
                text_surf.get_rect(midright=(self.ejex-11, y_cord))
            )
            if line == 5:
                ejey_title = pc_constant_font.render("Posición [cm]", True, 'white')
                ejey_title = pygame.transform.rotate(ejey_title, 90.0)
                display_surface.blit(
                    ejey_title,
                    ejey_title.get_rect(midright=(self.ejex-60, y_cord))
                )
        if len(self.datos) > self.divisiones:
            self.datos.pop(0)
        pixeles = []
        for line in range(self.divisiones): # eje x
            x_cord = map_range(line, 0, 9, self.ejex, self.ejex+self.cero)
            pygame.draw.line(display_surface, 'white', (x_cord, self.cero), (x_cord, self.cero+10), width=1)
            if line < len(self.datos):
                text_surf = ball_position_font.render(str(round(self.datos[line][0])), True, 'white')
                text_surf = pygame.transform.rotate(text_surf, 335.0)
                display_surface.blit(
                    text_surf,
                    text_surf.get_rect(midtop=(x_cord, self.cero+11))
                )
                pixeles.append((x_cord, self.datos[line][1]))
        ejex_title = pc_constant_font.render("Tiempo [seg]", True, 'white')
        display_surface.blit(
            ejex_title,
            ejex_title.get_rect(midtop=(self.ejex+self.cero-self.cero//2, self.cero+100))
        )
        if len(self.datos) < 2:
            pass
        else:
            pygame.draw.lines(display_surface, (2, 131, 230), False, pixeles, width=2)

def map_range(value, in_min, in_max, out_min, out_max):
  return (value - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

WINDOW_WIDTH = 1240
WINDOW_HEIGHT = 720
# Version de IA
while True:
    version_modelo = input("Ingrese una versión de modelo [1, 2, 3, 4, 5]\n: ")
    if version_modelo in ("1", "2", "3", "4", "5"):
        break
    print("La versión debe ser 1, 2, 3, 4, 5...")
#
pygame.init()
display_surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption(title="Modelo Ventilador")
clock = pygame.time.Clock()

# CIRCLES
TOP = 60
POS = 20
SENSOR_YPOS = 35
SENSOR_IZQ = POS/8
SENSOR_DER = POS/2.5
pc_rect = pygame.Rect(50, WINDOW_HEIGHT-150, 450, 150)
top_circle = pygame.Rect(pc_rect.centerx-POS, TOP-POS//2, (POS+1)*2, POS)
sensor_one = pygame.Rect(pc_rect.centerx-SENSOR_DER+1, SENSOR_YPOS+2.5, SENSOR_DER-SENSOR_IZQ, 5)
sensor_two = pygame.Rect(pc_rect.centerx+SENSOR_IZQ+1, SENSOR_YPOS+2.5, SENSOR_DER-SENSOR_IZQ, 5)
constant_font = pygame.font.Font(pygame.font.match_font('calibri'), 18)
pc_constant_font = pygame.font.Font(pygame.font.match_font('calibri'), 27)
ball_position_font = pygame.font.Font(pygame.font.match_font('calibri'), 20)
goku_image = pygame.image.load('images/goku.png').convert_alpha()
goku_image_scaled = pygame.transform.scale(goku_image, pygame.math.Vector2(goku_image.get_size())*0.1)
goku_rect = goku_image_scaled.get_rect(midbottom=pc_rect.midtop)

# TIMER
simulation_timer = pygame.event.custom_type()
pygame.time.set_timer(simulation_timer, 1000) # cada un segundo

posicion = 0
presion = 0
referencia = 50
ref_surf = ball_position_font.render("Ref", True, (2, 131, 230))
grafico = Grafico()
grafico.update(0, 0)


#############
# Cargando IA
#############
modelo: tf.keras.Sequential = tf.keras.models.load_model(f"archivos_generados/ventilador_red_neuronal{version_modelo}.keras")
###### USANDO MODELO IA Y SIMULACION CAJA NEGRA ######
presion = modelo.predict([[referencia/100]], verbose=False)[0][0]
#################
# Cargando Planta
#################
planta = Simulacion.Planta(
    pelota=Simulacion.Pelota(),
    tubo=Simulacion.Tubo(),
    ventilador=Simulacion.Ventilador()
    )
corutina = planta()
corutina.send(None)
#################
# ###############
#################
def ingresar_posicion():
    global referencia
    global presion
    while True:
        try:
            posicion_nueva = int(input("Ingrese una posición de [0-100]cm\n: "))
            if posicion_nueva < 0 or posicion_nueva > 100:
                raise PosicionFueraDeRangoError("Fuera de rango\n")
            referencia = posicion_nueva
            ###### USANDO MODELO IA Y SIMULACION CAJA NEGRA ######
            presion = modelo.predict([[referencia/100]], verbose=False)[0][0]
        except (ValueError, PosicionFueraDeRangoError):
            print("Por favor ingrese una posición válida para la planta...")

hilo = threading.Thread(target=ingresar_posicion)
hilo.start()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            pid = os.getpid()
            os.kill(pid, signal.SIGTERM)
        if event.type == simulation_timer:
            for x in range(200):
                posicion, tiempo, _, _, _ = corutina.send(presion)
            posicion = posicion * 100
            grafico.update(posicion, tiempo)
        if event.type == pygame.KEYDOWN:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                referencia += 1
                ###### USANDO MODELO IA Y SIMULACION CAJA NEGRA ######
                presion = modelo.predict([[referencia/100]], verbose=False)[0][0]
            if keys[pygame.K_DOWN]:
                referencia -= 1
                ###### USANDO MODELO IA Y SIMULACION CAJA NEGRA ######
                presion = modelo.predict([[referencia/100]], verbose=False)[0][0]
    if referencia > 100:
        referencia = 100
    if referencia < 0:
        referencia = 0
    dt = clock.tick() / 1000
    display_surface.fill((0,0,0))
    ref_cord = map_range(referencia, 0, 100, goku_rect.top-POS*2, TOP)
    pygame.draw.line(display_surface, (2, 131, 230), (pc_rect.centerx+POS, ref_cord), (pc_rect.centerx+POS+10, ref_cord), width=1)
    display_surface.blit(
            ref_surf,
            ref_surf.get_rect(midleft=(pc_rect.centerx+POS+10+1, ref_cord))
        )
    pygame.draw.circle(display_surface, 'white', (pc_rect.centerx, map_range(posicion, 0, 100, goku_rect.top-POS, TOP+POS)), POS-1)
    pygame.draw.lines(display_surface, (167, 181, 177), True, ((pc_rect.centerx-POS, TOP),(pc_rect.centerx-POS, pc_rect.top)), width=2)
    pygame.draw.lines(display_surface, (167, 181, 177), True, ((pc_rect.centerx+POS, TOP),(pc_rect.centerx+POS, pc_rect.top)), width=2)
    pygame.draw.ellipse(display_surface, (167, 181, 177), top_circle, width=2)
    pygame.draw.rect(display_surface, (2, 131, 230), pc_rect, width=5, border_radius=2)
    pygame.draw.line(display_surface, 'white', (pc_rect.centerx-POS, TOP), (pc_rect.centerx-POS-10, TOP), width=1)
    pygame.draw.line(display_surface, 'white', (pc_rect.centerx-POS, goku_rect.top-POS*2), (pc_rect.centerx-POS-10, goku_rect.top-POS*2), width=1)
    for index, texto in enumerate((("100", (pc_rect.centerx-POS-10, TOP)), ("0", (pc_rect.centerx-POS-10, goku_rect.top-POS*2)))):
        text_surf = constant_font.render(texto[0], True, 'white')
        text_rect = text_surf.get_rect(midright=texto[1])
        display_surface.blit(text_surf, text_rect)
    goku_image_scaled_copy = goku_image_scaled.copy()
    goku_image_scaled_copy.set_alpha(map_range(presion, 0, 255, 20, 255))
    display_surface.blit(goku_image_scaled_copy, goku_rect)
    for index, texto in enumerate((f"Largo del Tubo: 100 cm", "Radio de la pelota: 10 cm")):
        text_surf = constant_font.render(texto, True, 'white')
        text_rect = text_surf.get_rect(topleft=(5, index*20+50))
        display_surface.blit(text_surf, text_rect)

    for index, texto in enumerate((f"Referencia: {referencia}", f"Posición: {round(posicion,2)}", f"Presión: {presion}")):
        text_surf = pc_constant_font.render(texto, True, 'white')
        display_surface.blit(
            text_surf,
            text_surf.get_rect(center=(pc_rect.centerx, pc_rect.top+33*(index+1)))
        )

    pygame.draw.line(display_surface, 'white', (pc_rect.centerx-POS/2, SENSOR_YPOS), (pc_rect.centerx+POS/2, SENSOR_YPOS))
    pygame.draw.line(display_surface, 'white', (pc_rect.centerx-SENSOR_DER, SENSOR_YPOS), (pc_rect.centerx-SENSOR_DER, SENSOR_YPOS+5))
    pygame.draw.line(display_surface, 'white', (pc_rect.centerx-SENSOR_IZQ+1, SENSOR_YPOS), (pc_rect.centerx-SENSOR_IZQ+1, SENSOR_YPOS+5))
    pygame.draw.ellipse(display_surface, (0, 255, 34), sensor_one)
    pygame.draw.line(display_surface, 'white', (pc_rect.centerx+SENSOR_DER, SENSOR_YPOS), (pc_rect.centerx+SENSOR_DER, SENSOR_YPOS+5))
    pygame.draw.line(display_surface, 'white', (pc_rect.centerx+SENSOR_IZQ, SENSOR_YPOS), (pc_rect.centerx+SENSOR_IZQ, SENSOR_YPOS+5))
    pygame.draw.ellipse(display_surface, (0, 255, 34), sensor_two)
    grafico.mostrar()

    pygame.display.update()