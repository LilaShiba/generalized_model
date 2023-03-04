# Load the classifier for face detection
face_detector = dlib.get_frontal_face_detector()

# Load the predictor for facial landmarks
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detector(gray, 1)

    # Loop over the faces
    for face in faces:
        # Get the facial landmarks for the face
        landmarks = predictor(gray, face)

        # Draw circles at the eye landmarks
        cv2.circle(frame, (landmarks.part(36).x, landmarks.part(36).y), 3, (0, 0, 255), -1)
        cv2.circle(frame, (landmarks.part(36).x, landmarks.part(37).y), 3, (0, 0, 255), -1)

        cv2.circle(frame, (landmarks.part(36).x, landmarks.part(45).y), 3, (0, 0, 255), -1)
        cv2.circle(frame, (landmarks.part(45).x, landmarks.part(46).y), 3, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()