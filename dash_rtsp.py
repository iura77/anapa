import dash
from dash import dcc, html
from ultralytics import YOLO
from flask import Flask, Response
import cv2

rtsp_addr = r'rtsp://admin:A1234567@188.170.176.190:8029/Streaming/Channels/102?transportmode=unicast&profile=Profile_1'
model_addr = 'best.pt'


cap = cv2.VideoCapture(rtsp_addr)
cap.set(cv2.CAP_PROP_FPS, 5)

model = YOLO(model_addr)


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
    html.H1("Webcam Test"),
    html.Img(src="/video_feed")
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
