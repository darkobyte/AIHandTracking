import numpy as np
import torch
import cv2
import os
import sys

from blazebase import resize_pad, denormalize_detections
from blazepalm import BlazePalm
from blazehand_landmark import BlazeHandLandmark
from visualization import draw_detections, draw_landmarks, HAND_CONNECTIONS

# Create output directory if it doesn't exist
output_dir = "handpic"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize GPU/CPU device
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# Initialize detectors
palm_detector = BlazePalm().to(gpu)
palm_detector.load_weights("blazepalm.pth")
palm_detector.load_anchors("anchors_palm.npy")
palm_detector.min_score_thresh = 0.75

hand_regressor = BlazeHandLandmark().to(gpu)
hand_regressor.load_weights("blazehand_landmark.pth")

# Initialize camera
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Error: Could not open camera")
    sys.exit(1)

WINDOW = "Hand Capture"
cv2.namedWindow(WINDOW)

image_count = 0
frame_delay = 3  # Reduced delay between captures
frame_counter = 0

print("Fast auto-capturing 30 hand images...")
print("Press ESC to exit")

while image_count < 30:
    ret, frame = capture.read()
    if not ret:
        break

    # Mirror image
    frame = np.ascontiguousarray(frame[:, ::-1, ::-1])

    # Detect palms
    img1, img2, scale, pad = resize_pad(frame)
    normalized_palm_detections = palm_detector.predict_on_image(img1)
    palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

    # Get hand landmarks
    if len(palm_detections) > 0:
        xc, yc, scale, theta = palm_detector.detection2roi(palm_detections.cpu())
        img, affine, box = hand_regressor.extract_roi(frame, xc, yc, theta, scale)
        flags, handed, landmarks = hand_regressor(img.to(gpu))

        # Create a clean copy for saving
        clean_frame = frame.copy()

        # Draw landmarks and detections only on display frame
        hand_detected = False
        num_hands = 0
        for i in range(len(flags)):
            if flags[i] > 0.5:
                hand_detected = True
                num_hands += 1
                landmark, flag = landmarks[i], flags[i]
                draw_landmarks(frame, landmark[:, :2], HAND_CONNECTIONS, size=2)

        draw_detections(frame, palm_detections)

        # Auto capture when hand is detected and delay has passed
        # Save the clean frame without landmarks
        if hand_detected and frame_counter >= frame_delay:
            for hand_idx in range(min(num_hands, 30 - image_count)):
                filename = os.path.join(
                    output_dir, f"hand_{image_count + 1:03d}.jpg"
                )  # Changed from image_count to image_count + 1
                cv2.imwrite(filename, clean_frame[:, :, ::-1])
                print(f"Saved {filename} ({image_count + 1}/30)")
                image_count += 1
                if image_count >= 30:
                    break
            frame_counter = 0

        # Show status
        cv2.putText(
            frame,
            f"Captured: {image_count}/30",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    # Display frame
    cv2.imshow(WINDOW, frame[:, :, ::-1])

    # Handle ESC key to exit
    if cv2.waitKey(1) == 27:
        break

    frame_counter += 1

capture.release()
cv2.destroyAllWindows()
print(f"\nCapture complete! Saved {image_count} hand images in '{output_dir}' folder.")
