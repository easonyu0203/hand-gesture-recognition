import asyncio
import cv2


class VideoInput:
    """
    This class opens the video input device and reads the frames continuously.
    you can use get frame to get frame
    """
    def __init__(self, device, maxsize=1):
        self.cap = cv2.VideoCapture(device)
        self.is_running = False
        self.frame_event = asyncio.Event()
        self.frame_queue = asyncio.Queue(maxsize=maxsize)

    async def start(self):
        self.is_running = True
        while self.is_running:
            success, image = self.cap.read()
            if not success:
                continue

            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)

            # Put the processed frame into the queue
            try:
                self.frame_queue.put_nowait(image)
            except asyncio.QueueFull:
                # If the queue is full, overwrite the oldest frame with the most recent frame
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(image)

            self.frame_event.set()
            # Run the event loop
            await asyncio.sleep(0)

    def stop(self):
        self.is_running = False
        cv2.destroyAllWindows()

    async def get_frame(self):
        # Wait for a new frame to be available
        await self.frame_event.wait()
        self.frame_event.clear()

        # Get the next frame from the queue
        frame = await self.frame_queue.get()
        self.frame_queue.task_done()

        return frame
