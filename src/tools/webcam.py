# MODULES (EXTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
import cv2
import numpy as np
from src.models.cnn import build_model
# ---------------------------------------------------------------------------------------------------------------------

# MODULES (INTERNAL)
# ---------------------------------------------------------------------------------------------------------------------
from common.constants import IMAGE_SIZE, EMOTION_LABELS, MODEL_PATH
# ---------------------------------------------------------------------------------------------------------------------

# OPERATIONS / CLASS CREATION / GENERAL FUNCTIONS
# ---------------------------------------------------------------------------------------------------------------------

__all__ = ['run_webcam']

CASCADE = 'assets/haarcascade_frontalface_default.xml'
"""
Path to the Haar Cascade classifier used for face detection.

This file contains a pre-trained model based on Haar features, used to locate human faces in grayscale images.
It is used for its speed and low computational cost, which makes it suitable for real-time applications such as 
emotion detection via webcam.

The classifier is applied as a preliminary step to emotion recognition, allowing the facial region to be extracted, 
which will then be processed by the convolutional neural network.
"""

COLOR = (0, 255, 0)
"""
Color used to draw visual elements on the captured image.
"""

def run_webcam():
    """
    Runs a real-time emotion detection and classification system using the webcam.

    This function initializes a previously trained neural network model and loads its weights
    from disk. It then accesses the system's webcam to capture real-time video and detect faces 
    using an OpenCV cascade classifier.

    **Processing flow:**
        - The model is built and the stored weights are loaded.
        - The face classifier (`CascadeClassifier`) is initialized.
        - The webcam is accessed using `cv2.VideoCapture`.
        - For each captured frame:
            * The image is converted to grayscale.
            * The faces present in the frame are detected.
            * Each detected face is:
                - Cropped.
                - Resized to the size defined in `IMAGE_SIZE`.
                - Normalized to the range [0, 1].
                - Restructured to fit the shape expected by the model.
            * The emotion is predicted.
            * A rectangle is drawn around the face and the label of the predicted emotion is displayed.
        - The video is displayed in a window in real time.

    **Interaction:**
        - Execution continues until the user presses the `q` key.
        - Upon completion, the webcam and OpenCV window resources are properly released. 

    This function is designed to run as an interactive, blocking process, 
    suitable for demonstrations or real-time testing.
    """
    model = build_model()
    model.load_weights(filepath=MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(CASCADE)

    cap = cv2.VideoCapture(index=0)

    if not cap.isOpened():
        print("The webcam could not be accessed")
        return
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            image=gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]

            face = cv2.resize(src=face, dsize=IMAGE_SIZE)
            face = face / 255.0
            face = np.reshape(face, (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))

            preds = model.predict(face, verbose=0)
            emotion = EMOTION_LABELS[np.argmax(preds)]

            cv2.rectangle(
                img=frame, 
                pt1=(x, y), 
                pt2=(x+w, y+h), 
                color=COLOR, 
                thickness=2
            )

            cv2.putText(
                img=frame,
                text=emotion,
                org=(x, y - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.9,
                color=COLOR,
                thickness=2
            )

        cv2.imshow("Emotion detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------------------------------------------------------------------------------------------------
# END OF FILE