import cv2
import argparse


def main(image_path):
    # Load image and turn into grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use the Haar Cascade classifier to detect faces
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    faces = face_cascade.detectMultiScale(gray)

    # Draw a rectangle around all found faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    # Show the result in a UI window
    cv2.imshow(f"Output of face detection for {image_path}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Face detection example")
    parser.add_argument("--image", help="Path to image.", default="philipp.jpg")
    args = parser.parse_args()

    main(args.image)
