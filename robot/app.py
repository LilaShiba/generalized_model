import RPi.GPIO as GPIO
from flask import Flask, render_template, Response
import subprocess
import picamera
import io
import time

app = Flask(__name__)
#GPIO.setmode(GPIO.BOARD)
PIR_PIN = 26
#GPIO.setup(PIR_PIN, GPIO.IN)

# Set up the GPIO pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.framerate = 30
        camera.start_preview()
        while True:
            frame = get_frame(camera)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
def generate_frames():
    with picamera.PiCamera() as camera:
        while True:
            stream = io.BytesIO()
            camera.capture(stream, format='jpeg')
            stream.seek(0)
            yield stream.read()
            time.sleep(0.1)

def get_frame(camera):
    stream = io.BytesIO()
    camera.capture(stream, format='jpeg', use_video_port=True)
    frame = stream.getvalue()
    stream.seek(0)
    stream.truncate()
    return frame

@app.route('/video_feed')
def video_feed():
    #return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/sensor_data')
def sensor_data():
    # Wait for the sensor to settle
    print("Waiting for sensor to settle...")
    time.sleep(2)
    print("Ready")

    try:
        while True:
            # Read the sensor value
            pir_value = GPIO.input(PIR_PIN)

            # Print the sensor value
            if pir_value:
                print("Motion detected!")
            else:
                print("No motion")

            # Wait for a short time before reading the sensor again
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Exiting program...")
    finally:
        # Clean up the GPIO pins
        GPIO.cleanup()

@app.route('/run_agent',methods=['POST'])
def run_agent():
    script_output = subprocess.check_output(['python', 'scripts/test.py'])
    return render_template('result.html', output=script_output)


    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)