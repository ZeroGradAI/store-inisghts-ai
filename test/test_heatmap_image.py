import cv2
from ultralytics import solutions

# Initialize the heatmap solution with the same parameters as the video version
heatmap = solutions.Heatmap(
    colormap=cv2.COLORMAP_PARULA,
    show=True,
    model="yolo11n.pt"
)

# Read the input image
# Replace 'path_to_your_image.jpg' with your actual image path
image_path = "store_image.jpg"  # Update this path to your image
im0 = cv2.imread(image_path)

if im0 is None:
    print(f"Error: Could not read image from {image_path}")
    exit(1)

# Process the image with heatmap
results = heatmap(im0)

# Wait for a key press and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows() 