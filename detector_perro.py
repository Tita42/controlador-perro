#!/usr/bin/env python3
"""
Detector de perro con webcam para Raspberry Pi.

Usa MobileNet-SSD (Caffe) con el modulo DNN de OpenCV: es chico (~23 MB),
corre en CPU a varios FPS en una Raspberry Pi 4/5 y detecta la clase "dog".
Cuando el perro aparece en varios frames seguidos reproduce un sonido,
y espera un tiempo (cooldown) antes de volver a sonar.

Ademas publica el video con las detecciones como stream MJPEG en
http://<raspberry>:8000/ (se ve desde cualquier navegador de la red).

Uso:
    python3 detector_perro.py --sonido ladrido.wav
    python3 detector_perro.py --sonido hola.mp3 --mostrar   # con ventana de video
    python3 detector_perro.py --puerto 0                    # sin servidor web
"""

import argparse
import os
import platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
ID_PERRO = CLASES.index("gato")


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


PAGINA_HTML = b"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Detector de perra</title>
<style>
  body { margin: 0; background: #111; color: #eee; font-family: sans-serif;
         display: flex; flex-direction: column; align-items: center; }
  h1 { font-size: 1.2rem; margin: 12px; }
  img { width: 100%; max-width: 960px; height: auto; }
  a { color: #8cf; margin: 10px; }
</style>
</head>
<body>
<h1>Detector de perra</h1>
<img src="/stream" alt="video en vivo">
<a href="/foto.jpg" download>Descargar foto</a>
</body>
</html>
"""


def ip_local():
    """IP de la Raspberry en la red local (no manda ningun paquete)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "localhost"


class _ManejadorHTTP(BaseHTTPRequestHandler):
    def do_GET(self):
        video = self.server.video
        if self.path == "/":
            self._responder(200, "text/html; charset=utf-8", PAGINA_HTML)
        elif self.path == "/foto.jpg":
            jpeg = video.jpeg
            if jpeg is None:
                self._responder(503, "text/plain", b"Todavia no hay imagen")
            else:
                self._responder(200, "image/jpeg", jpeg)
        elif self.path == "/stream":
            self._stream(video)
        else:
            self._responder(404, "text/plain", b"No encontrado")

    def _responder(self, codigo, tipo, cuerpo):
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(cuerpo)

    def _stream(self, video):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        jpeg = None
        try:
            while True:
                jpeg = video.siguiente(jpeg)
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: %d\r\n\r\n" % len(jpeg))
                self.wfile.write(jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # el navegador se desconecto

    def log_message(self, *args):
        pass  # no llenar la consola con cada pedido


class ServidorVideo:
    """Publica el ultimo frame como stream MJPEG en un hilo aparte."""

    def __init__(self, puerto):
        self.jpeg = None
        self.cond = threading.Condition()
        try:
            self.httpd = ThreadingHTTPServer(("0.0.0.0", puerto), _ManejadorHTTP)
        except OSError as e:
            sys.exit(f"No pude abrir el puerto {puerto}: {e}")
        self.httpd.daemon_threads = True
        self.httpd.video = self
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        print(f"Video en http://{ip_local()}:{puerto}/")

    def publicar(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.cond:
                self.jpeg = buf.tobytes()
                self.cond.notify_all()

    def siguiente(self, anterior):
        """Espera un frame distinto de `anterior` (o hasta 5 s) y lo devuelve."""
        with self.cond:
            self.cond.wait_for(lambda: self.jpeg is not anterior, timeout=5)
            return self.jpeg

    def cerrar(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def dibujar(frame, perros):
    for confianza, (x1, y1, x2, y2) in perros:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"perro {confianza:.0%}", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"), (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


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
    p.add_argument("--puerto", type=int, default=8000,
                   help="puerto del servidor web con el video (default 8000, 0 = apagado)")
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

    video = ServidorVideo(args.puerto) if args.puerto else None

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

            if args.mostrar or video:
                dibujar(frame, perros)
            if video:
                video.publicar(frame)
            if args.mostrar:
                cv2.imshow("detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        if video:
            video.cerrar()
        if args.mostrar:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
