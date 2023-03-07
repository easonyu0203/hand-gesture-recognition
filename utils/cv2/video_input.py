import cv2


class VideoInput:
    """
    This class opens the video input device and reads the frames continuously.
    It emits an event on_image_processed whenever a new frame is available for processing.
    """
    def __init__(self, device):
        self.cap = cv2.VideoCapture(device)
        self.is_running = False

    def start(self):
        self.is_running = True
        while self.is_running:
            success, image = self.cap.read()
            if not success:
                continue

            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)
            # emit event for image processing
            self.on_image_processed(image)

    def stop(self):
        self.is_running = False

    def on_image_processed(self, image):
        pass
