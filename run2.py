import dash
from dash import dcc, html, Input, Output, ctx, callback, dcc
from ultralytics import YOLO
from flask import Flask, Response
import cv2
import urllib.request

##rtsp_addr = r'rtsp://admin:A1234567@188.170.176.190:8029/Streaming/Channels/102?transportmode=unicast&profile=Profile_1'

rtsp_addr = ''
model_addr = r'./best.pt'


#cap = cv2.VideoCapture(rtsp_addr)
#cap.set(cv2.CAP_PROP_FPS, 5)

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

@server.route('/video_get')
def video_get():
    if rtsp_addr != '':
        response = Response(gen(cap), mimetype='multipart/x-mixed-replace; boundary=frame')
        return response
    else:
        return Response('signal_lose.png', mimetype='image/jpg')

app.layout = html.Div([
    html.H1("Webcam вернули с сервера"),
    html.Img(src="{% response %}")
])

@server.route('/video_feed')
def video_feed():
    if rtsp_addr != '':
        return Response(gen(cap), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response('signal_lose.png', mimetype='image/jpg')

app.layout = html.Div([
    html.H1("Введите адрес RTSP потока: "),
    dcc.Input(id='my-input', type='url', debounce=True),
    html.Button('OK', id='btn-nclicks-1', n_clicks=0),
    html.Br(),
    html.Div(id='container-button-timestamp'),
    html.Br(),
    html.Img(src="/video_get")
])

@app.callback(
    Output('container-button-timestamp', 'children'),
    Input('btn-nclicks-1', 'n_clicks'),
    Input('my-input', 'value'),
)
def displayClick(btn1, inpt):
    msg = "Не введён поток RTSP"
    if "btn-nclicks-1" == ctx.triggered_id:
        global cap, rstp_addr
        msg = inpt
        rstp_addr = inpt
        cap = cv2.VideoCapture(inpt)
        video_get()
    return msg


if __name__ == '__main__':
    app.run_server(debug=True, port=8880)
