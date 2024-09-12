import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

# Reading the image and the template
img = cv2.imread('testimages\page_9.png')
temp = cv2.imread('cropped-ad.png')

if img is None:
    print("Error: Could not read 'testimages/post_9.png'")
    exit()
if temp is None:
    print("Error: Could not read 'cropped-ad.png'")
    exit()

# Save the image dimensions
W, H = temp.shape[:2]

# Define a minimum threshold
thresh = 0.4

# Converting them to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
temp_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

# Passing the image to matchTemplate method
match = cv2.matchTemplate(image=img_gray, templ=temp_gray, method=cv2.TM_CCOEFF_NORMED)

# Select rectangles with confidence greater than threshold
(y_points, x_points) = np.where(match >= thresh)

# Initialize our list of rectangles
boxes = []

# Loop over the starting (x, y)-coordinates
for (x, y) in zip(x_points, y_points):
    # Update our list of rectangles
    boxes.append((x, y, x + W, y + H))

# Apply non-maxima suppression to the rectangles
# This will create a single bounding box
boxes = non_max_suppression(np.array(boxes))

# Loop over the final bounding boxes
for (x1, y1, x2, y2) in boxes:
    # Draw the bounding box on the image
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

# Show the template and the final output
cv2.imshow("Template", temp)
cv2.imshow("After NMS", img)
cv2.waitKey(0)

# Destroy all the windows manually to be on the safe side
cv2.destroyAllWindows()
