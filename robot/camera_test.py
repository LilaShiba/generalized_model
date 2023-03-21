import cv2
import time

# initialize camera
camera = cv2.VideoCapture(0)

# set camera resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# allow camera to warm up
time.sleep(2)

while True:
    # capture frame-by-frame
    ret, frame = camera.read()

    # flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # write the flipped frame to video file
    out.write(frame)

    # display the resulting frame
    cv2.imshow('frame', frame)

    # wait for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera and video writer
camera.release()
out.release()

# destroy all windows
cv2.destroyAllWindows(