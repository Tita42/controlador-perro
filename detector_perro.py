#!/usr/bin/env python3
"""
Detector de perro con webcam para Raspberry Pi.

Usa MobileNet-SSD (Caffe) con el modulo DNN de OpenCV: es chico (~23 MB),
corre en CPU a varios FPS en una Raspberry Pi 4/5 y detecta la clase "dog".
Cuando el perro aparece en varios frames seguidos reproduce un sonido,
y espera un tiempo (cooldown) antes de volver a sonar.

Uso:
    python3 detector_perro.py --sonido ladrido.wav
    python3 detector_perro.py --sonido hola.mp3 --mostrar   # con ventana de video
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

import cv2

DIR_BASE = os.path.dirname(os.path.abspath(__file__))
DIR_MODELOS = os.path.join(DIR_BASE, "modelos")
# Sonido por defecto si no se pasa --sonido
SONIDO_DEFAULT = os.path.join(DIR_BASE, "sonidos", "alarma-sismo.mp3")

URL_REPO = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/"
PROTOTXT = os.path.join(DIR_MODELOS, "deploy.prototxt")
CAFFEMODEL = os.path.join(DIR_MODELOS, "mobilenet_iter_73000.caffemodel")

# Clases del dataset VOC con las que fue entrenado MobileNet-SSD
CLASES = ["fondo", "avion", "bicicleta", "pajaro", "bote", "botella", "bus",
          "auto", "gato", "silla", "vaca", "mesa", "perro", "caballo", "moto",
          "persona", "planta", "oveja", "sofa", "tren", "monitor"]
ID_PERRO = CLASES.index("perro")


def descargar_modelo():
    os.makedirs(DIR_MODELOS, exist_ok=True)
    for ruta in (PROTOTXT, CAFFEMODEL):
        if not os.path.exists(ruta):
            url = URL_REPO + os.path.basename(ruta)
            print(f"Descargando {url} ...")
            urllib.request.urlretrieve(url, ruta)


class Reproductor:
    """Reproduce un archivo de audio sin bloquear el loop de video."""

    def __init__(self, archivo):
        if not os.path.exists(archivo):
            sys.exit(f"No existe el archivo de sonido: {archivo}")
        self.archivo = archivo
        self.comando = self._elegir_comando(archivo)
        self.proceso = None

    @staticmethod
    def _elegir_comando(archivo):
        if platform.system() == "Darwin": #esto es el reproductor de audio para audio de python en mac
            return ["afplay"]
        candidatos = [
            ["mpg123", "-q"],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
            ["cvlc", "--play-and-exit", "--quiet"],
        ]
        if archivo.lower().endswith(".wav"):
            candidatos.insert(0, ["aplay", "-q"])
        for cmd in candidatos:
            if shutil.which(cmd[0]):
                return cmd
        sys.exit("No se encontró el sonido")

    def sonando(self):
        return self.proceso is not None and self.proceso.poll() is None

    def reproducir(self):
        if not self.sonando():
            self.proceso = subprocess.Popen(self.comando + [self.archivo])


def detectar_perros(red, frame, confianza_min):
    """Devuelve una lista de (confianza, (x1, y1, x2, y2)) de los perros en el frame."""
    alto, ancho = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    red.setInput(blob)
    salida = red.forward()  # forma: (1, 1, N, 7)

    perros = []
    for det in salida[0, 0]:
        clase, confianza = int(det[1]), float(det[2])
        if clase == ID_PERRO and confianza >= confianza_min:
            x1, y1, x2, y2 = (det[3:7] * [ancho, alto, ancho, alto]).astype(int)
            perros.append((confianza, (x1, y1, x2, y2)))
    return perros


def main():
    p = argparse.ArgumentParser(description="Detecta a tu perra y reproduce un sonido")
    p.add_argument("--sonido", default=SONIDO_DEFAULT,
                   help="archivo .wav o .mp3 a reproducir (default sonidos/alarma-sismo.mp3)")
    p.add_argument("--camara", type=int, default=0, help="indice de la webcam (default 0)")
    p.add_argument("--confianza", type=float, default=0.5,
                   help="confianza minima de la deteccion, 0-1 (default 0.5)")
    p.add_argument("--umbral-seguidos", type=float, default=0.85,
                   help="confianza minima para que el frame cuente como seguido, 0-1 (default 0.85)")
    p.add_argument("--frames-confirmacion", type=int, default=3,
                   help="frames seguidos con perro antes de sonar (evita falsos positivos)")
    p.add_argument("--cooldown", type=float, default=15,
                   help="segundos minimos entre sonidos (default 15)")
    p.add_argument("--mostrar", action="store_true",
                   help="muestra ventana con el video (necesita pantalla)")
    args = p.parse_args()

    descargar_modelo()
    red = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
    reproductor = Reproductor(args.sonido)

    cam = cv2.VideoCapture(args.camara)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # menos latencia: siempre el frame mas nuevo
    if not cam.isOpened():
        sys.exit(f"No pude abrir la camara {args.camara}")

    print("Vigilando... (Ctrl+C para salir)")
    seguidos = 0
    ultimo_sonido = 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("No pude leer un frame, reintentando...")
                time.sleep(0.5)
                continue

            perros = detectar_perros(red, frame, args.confianza)
            # Solo cuenta el frame si algun perro supera el umbral de seguidos
            perro_seguro = any(c >= args.umbral_seguidos for c, _ in perros)
            seguidos = seguidos + 1 if perro_seguro else 0

            ahora = time.time()
            if (seguidos >= args.frames_confirmacion
                    and ahora - ultimo_sonido >= args.cooldown):
                mejor = max(c for c, _ in perros)
                print(f"[{time.strftime('%H:%M:%S')}] Perra detectada ({mejor:.0%}) -> sonido")
                reproductor.reproducir()
                ultimo_sonido = ahora

            if args.mostrar:
                for confianza, (x1, y1, x2, y2) in perros:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"perro {confianza:.0%}", (x1, max(y1 - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        if args.mostrar:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
