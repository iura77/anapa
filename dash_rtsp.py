import dash
from dash import dcc, html
from ultralytics import YOLO
from flask import Flask, Response
import cv2

# введите адрес видеопотока в переменную rtsp_addr
rtsp_addr = r'rtsp://admin:A1234567@188.170.176.190:8029/Streaming/Channels/102?transportmode=unicast&profile=Profile_1


# путь к модели
model_addr = 'best.pt'

# открываеам видеопоток
cap = cv2.VideoCapture(rtsp_addr)

# загружаем модель
model = YOLO(model_addr)

# функция возвращает поток распознанных кадров
def gen(camera):
    while True:
        success, img = camera.read()
        result = model.predict(img, verbose=False, conf=0.35)
        annotated_frame = result[0].plot(conf=False, labels=False)
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

@server.route('/video_feed')
def video_feed():
    return Response(gen(cap),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = html.Div([
    html.H1("CCTV cam vendor detection"),
    html.Img(src="/video_feed")
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
