import cv2
import time

def show_camera_feeds(max_index=4, display_time=5):
    print(f"Displaying each available camera feed for {display_time} seconds...")
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Camera {idx}: Opened successfully. Showing feed...")
            start_time = time.time()
            while time.time() - start_time < display_time:
                ret, frame = cap.read()
                if not ret:
                    print(f"Camera {idx}: Could not read frame.")
                    break
                cv2.imshow(f'Camera {idx}', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            print(f"Camera {idx}: Not available.")
    print("Done displaying all camera feeds.")

if __name__ == "__main__":
    show_camera_feeds(4, display_time=5) 