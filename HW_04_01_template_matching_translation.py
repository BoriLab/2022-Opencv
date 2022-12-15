import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Set the working directory to be the current one
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load a reference image as grayscale
img_reference = cv2.imread('test_0.png', 0)

# Load a template image as grayscale
img_template = cv2.imread('fish.png', 0)
w, h = img_template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED',
           'cv2.TM_CCORR',  'cv2.TM_CCORR_NORMED',
           'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']

for idx, meth in enumerate(methods, start=1):
    # Apply template matching
    method = eval(meth)
    img_matching_scores = cv2.matchTemplate(img_reference, img_template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img_matching_scores)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Draw a bounding box
    img_found = cv2.cvtColor(img_reference, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_found, top_left, bottom_right, [0, 0, 255], 2)
    
    plt.subplot(2, len(methods), idx),  plt.imshow(img_matching_scores, cmap = 'gray')
    plt.title(meth), plt.xticks([]), plt.yticks([])
    plt.subplot(2, len(methods), len(methods)+idx),  plt.imshow(cv2.cvtColor(img_found, cv2.COLOR_BGR2RGB))
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

plt.suptitle('Template Matching')
plt.show()