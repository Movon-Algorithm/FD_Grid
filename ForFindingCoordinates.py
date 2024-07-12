import cv2

# Callback function to capture the coordinates of mouse clicks
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print the coordinates on the console
        print(f"Coordinates: ({x}, {y})")
        # Draw a small circle on the frame where the user clicked
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        # Display the updated frame with the circle
        cv2.imshow("Video", frame)

# Capture video from file or webcam
video_path = 'input_video04.mp4'  # Use '0' for webcam
cap = cv2.VideoCapture(video_path)

# Check if the video capture is initialized
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create a named window
cv2.namedWindow("Video")

# Set the mouse callback function to capture coordinates
cv2.setMouseCallback("Video", get_coordinates)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("End of video stream or cannot receive frame.")
        break

    # Resize the frame to 1280x720
    frame = cv2.resize(frame, (1280, 720))

    # Display the resulting frame
    cv2.imshow("Video", frame)

    while True:
        # Wait for a key press and check if it's 'q' to break the loop
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif key == ord('n'):
            break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()


################# Writing coordinates
#(1347,378), (908,1040), (1695, 842), 
#(1737, 817), (1346, 324), (952,843)