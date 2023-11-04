import cv2

print("\n\n\n\n\n\n Loaded Tensorflow \n\n\n\n\n\n")

class camera:
  def __init__(self):
    self.cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not self.cap.isOpened():
        raise IOError("Cannot open webcam")

  def destroycamera(self):
    self.cap.release()
    cv2.destroyAllWindows()

  def shot(self):
    ret, frame = self.cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, 
    interpolation=cv2.INTER_AREA)
    frame = cv2.resize(frame, (256, 256), 
               interpolation = cv2.INTER_LINEAR)
    return frame



def main():
    myobj = camera()

    frame = myobj.shot()

    print(frame.shape)

    cv2.imshow("Image", frame)

    cv2.waitKey(0)

if __name__ == "__main__":
    main()