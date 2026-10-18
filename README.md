# Detector de perra

Detecta a la perra con una webcam y reproduce un sonido.

## Instalación en la Raspberry Pi (Raspberry Pi OS)

```bash
sudo apt update
sudo apt install python3-opencv mpg123 alsa-utils
```

(`python3-opencv` de apt es OpenCV 4.x; **no uses OpenCV 5**, que ya no soporta modelos Caffe.)

Copia esta carpeta a la Raspberry junto con tu sonido. La primera vez que corre, el script descarga el modelo (~23 MB) a `modelos/`.

## Instalación en tu compu (con venv)

`venv/` y `modelos/` no están en el repo. Después de clonar:

```bash
python3 -m venv venv
source venv/bin/activate        # en Windows: venv\Scripts\activate
pip install -r requirements.txt
python detector_perro.py --sonido sonidos/alarma-sismo.mp3   # descarga el modelo la primera vez
```

## Uso

```bash
python3 detector_perro.py --sonido mi_sonido.mp3
```

Opciones:

| opción | default | qué hace |
|---|---|---|
| `--confianza` | 0.5 | sube a 0.6–0.7 si hay falsos positivos |
| `--frames-confirmacion` | 3 | frames seguidos con perro antes de sonar |
| `--cooldown` | 15 | segundos mínimos entre sonidos |
| `--camara` | 0 | índice de la webcam |
| `--mostrar` | off | ventana con las detecciones (necesita pantalla; `q` para salir) |

## Que arranque solo al prender la Raspberry

Crea `/etc/systemd/system/detector-perro.service` (ajusta usuario y rutas):

```ini
[Unit]
Description=Detector de perra
After=sound.target

[Service]
User=pi
WorkingDirectory=/home/pi/gorda
ExecStart=/usr/bin/python3 /home/pi/gorda/detector_perro.py --sonido /home/pi/gorda/mi_sonido.mp3
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now detector-perro
journalctl -u detector-perro -f   # ver logs
```
