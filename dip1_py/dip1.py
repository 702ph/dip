from __future__ import annotations
import cv2
import numpy as np

def do_something_that_my_tutor_is_gonna_like(img: np.ndarray) -> np.ndarray:
    """
    Requirements:
      - Accepts np.ndarray (BGR if color).
      - Returns processed image (BGR or grayscale OK).
      - Must produce a result sufficiently different from input
    """
    if img is None or not hasattr(img, "ndim"):
        raise ValueError("Input image is invalid")
    # TODO: replace this placeholder with your pipeline
    # raise NotImplementedError("Implement your processing here")
    # img = cv2.bitwise_not(img)

    # h, w = img.shape[:2]
    
    # center = (w / 2, h / 2)

    # angle = 90
    
    # M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    
    # rotated = cv2.warpAffine(img, M, (w, h))
    
    img[:,:,2]= 255
    img[:,:,1]= 10
    img[:,:,0]= 10

    # print(img.shape)
    (img.shape)

    return img  # <-- temporary; replace with your result

    return img  # <-- temporary; replace with your result

def run(filename: str) -> None:
    """Load image, call processing, show and save."""
    win1 = "Original image"
    win2 = "Result"

    print("loading image")
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    print("done")

    if img is None:
        raise FileNotFoundError(f"ERROR: Cannot read file {filename}")

    cv2.namedWindow(win1)
    cv2.imshow(win1, img)

    out = do_something_that_my_tutor_is_gonna_like(img)

    cv2.namedWindow(win2)
    cv2.imshow(win2, out)

    cv2.imwrite("result.png", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
