import cv2
import glob
import numpy as np
from scipy.signal import wiener

"""
Utilities for filtering incoming images

"""
def check_blur(image, thresh=45):
    """
    If True is returned, is blurry.
    """
    # image = cv2.imread(img_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    meas = cv2.Laplacian(gray, cv2.CV_64F).var()
    #print(img_file, ", blurriness is ", meas)
    if meas<thresh:
        print("Blurry!")
        return True
    return False

def similar(file1, file2, thres = 120):
    """
    If True is returned, similar!
    """
    # file1 = cv2.imread(img_file1)
    # file2 = cv2.imread(img_file2)
    img1 = cv2.cvtColor(file1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(file2, cv2.COLOR_BGR2GRAY)

    
    #? ANOTHER: minHessian = 400
    #? detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    if desc1 is None or desc2 is None:
        print("Error!")
        return True
   
    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    nNeighbours = 2
    matches = flann.knnMatch(desc1, desc2, k=nNeighbours)
    goodMatches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            goodMatches.append(m)
    print("Matches between images: ", len(goodMatches))
    if len(goodMatches)>thres: return True
    return False
     
def apply_wiener_filter(image):
    """
    Apply Wiener filtering to reduce noise in the image.
    
    Args:
        image (np.ndarray): The input image.
    
    Returns:
        np.ndarray: The filtered image.
    """
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_image = wiener(gray_image, (5, 5))
    # Convert back to BGR
    filtered_image_bgr = cv2.cvtColor(filtered_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return filtered_image_bgr

def interpolate_frames(frame1, frame2, num_intermediate_frames=5):
    """
    Interpolate frames between two given frames.

    Args:
        frame1 (np.ndarray): The first frame.
        frame2 (np.ndarray): The second frame.
        num_intermediate_frames (int): Number of intermediate frames to generate.

    Returns:
        List[np.ndarray]: List of interpolated frames.
    """
    interpolated_frames = []
    for i in range(1, num_intermediate_frames + 1):
        alpha = i / (num_intermediate_frames + 1)
        interpolated_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        interpolated_frames.append(interpolated_frame)
    return interpolated_frames

if __name__=="__main__":
    files = sorted(glob.glob("temps/*.png"))
    to_save=[]
    # for file in files:
    #     ak = check_blur(file)
    current = 1
    to_save.append(files[0])
    while current<len(files):
        
        ch = check_blur(files[current])
        if ch:
            current+=1
            continue
        print("Comparing: ", to_save[-1], files[current])
        similar_ = similar(to_save[-1], files[current])
        
        if not similar_:
            to_save.append(files[current])
            current+=1
            continue
        current+=1
    print(to_save)

    



