
__version__ = "$Revision: 2018021201 $"

########################################################################
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from BlobProperties import BlobProperties

########################################################################

#<!--------------------------------------------------------------------------->
#<!--                               Functions                               -->
#<!--------------------------------------------------------------------------->
def getHomographyFromMouse(image1, image2, N=4):
    """
    getHomographyFromMouse(image1, image2, N=4) -> homography, mousePoints

    Calculates the homography from a plane in image "image1" to a plane in image "image2" by using the mouse to define corresponding points
    Returns: 3x3 homography matrix and a set of corresponding points used to define the homography
    Parameters: N >= 4 is the number of expected mouse points in each image,
                when N < 0: then the corners of image "image1" will be used as input and thus only 4 mouse clicks are needed in image "image2".

    Usage: Use left click to select a point and right click to remove the most recently selected point.
    """
    # Vector with all input images.
    images = []
    images.append(cv2.cvtColor(image1.copy(), cv2.COLOR_BGR2RGB))
    images.append(cv2.cvtColor(image2.copy(), cv2.COLOR_BGR2RGB))

    # Vector with the points selected in the input images.
    mousePoints = []

    # Control the number of processed images.
    firstImage = 0

    # When N < 0, then the corners of image "image1" will be used as input.
    if N < 0:

        # Force 4 points to be selected.
        N = 4
        firstImage = 1
        m, n = image1.shape[:2]

        # Define corner points from image "image1".
        mousePoints.append([(0, 0), (n, 0), (n, m), (0, m)])

    # Check if there is the minimum number of needed points to estimate the homography.
    if math.fabs(N) < 4:
        N = 4
        print("At least 4 points are needed!!!")

    # Make a pylab figure window.
    fig = plt.figure(1)

    # Get the correspoding points from the input images.
    for i in range(firstImage, 2):
        # Setup the pylab subplot.
        plt.subplot(1, 2, i+1)
        plt.imshow(images[i])
        plt.axis("image")
        plt.title("Click " + str(N) + " times in this image.")
        fig.canvas.draw()

        # Get mouse inputs.
        mousePoints.append(fig.ginput(N, -1))

        # Draw selected points in the processed image.
        for point in mousePoints[i]:
            cv2.circle(images[i], (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
        plt.imshow(images[i])
        fig.canvas.draw()

    # Close the pylab figure window.
    plt.close(fig)

    # Convert to OpenCV format.
    points1 = np.array([[x, y] for (x, y) in mousePoints[0]])
    points2 = np.array([[x, y] for (x, y) in mousePoints[1]])

    # Calculate the homography.
    homography, mask = cv2.findHomography(points1, points2)
    return homography, mousePoints

def estimateHomography(points1, points2):
    """
    estimateHomography(points1, points2) -> homography

    Calculates a homography from one plane to another using a set of 4 points from each plane.
    Returns: 3x3 homography matrix and a set of corresponding points used to define the homography
    Parameters: points1 is 4 points (np.array) on a plane
                points2 is the corresponding 4 points (np.array) on an other plane
    """
    # Check if there is sufficient points.
    if len(points1) == 4 and len(points2) == 4:
        # Get x, y values.
        x1, y1 = points1[0]
        x2, y2 = points1[1]
        x3, y3 = points1[2]
        x4, y4 = points1[3]

        # Get x tilde and y tilde values.
        x_1, y_1 = points2[0]
        x_2, y_2 = points2[1]
        x_3, y_3 = points2[2]
        x_4, y_4 = points2[3]

        # Create matrix A.
        A = np.array([[-x1, -y1, -1, 0, 0, 0, x1 * x_1, y1 * x_1, x_1],
                      [0, 0, 0, -x1, -y1, -1, x1 * y_1, y1 * y_1, y_1],
                      [-x2, -y2, -1, 0, 0, 0, x2 * x_2, y2 * x_2, x_2],
                      [0, 0, 0, -x2, -y2, -1, x2 * y_2, y2 * y_2, y_2],
                      [-x3, -y3, -1, 0, 0, 0, x3 * x_3, y3 * x_3, x_3],
                      [0, 0, 0, -x3, -y3, -1, x3 * y_3, y3 * y_3, y_3],
                      [-x4, -y4, -1, 0, 0, 0, x4 * x_4, y4 * x_4, x_4],
                      [0, 0, 0, -x4, -y4, -1, x4 * y_4, y4 * y_4, y_4]])

        # Calculate SVD.
        U, D, V = np.linalg.svd(A)

        # Get last row of V returned from SVD.
        h = V[8]

        # Create homography matrix.
        homography = np.array([[h[0], h[1], h[2]],
                               [h[3], h[4], h[5]],
                               [h[6], h[7], h[8]]])

        # Normalize homography.
        homography /= homography[2, 2]

        # Return the homography matrix.
        return homography

def getContourProperties(contour, properties=[]):
    """
    This function is used for getting descriptors of contour-based connected
    components.

    The main method to use is: getContourProperties(contour, properties=[]):
    contour: the contours variable found by cv2.findContours()
    properties: list of strings specifying which properties should be
                calculated and returned.

    The following properties can be specified:
    Approximation: A contour shape to another shape with less number of vertices.
    Area: Area within the contour - float.
    Boundingbox: Bounding box around contour - 4 tuple (topleft.x, topleft.y,
                                                        width, height).
    Centroid: The center of contour - (x, y).
    Circle: The the circumcircle of an object.
    Circularity: Used to check if the countour is a circle.
    Convexhull: Calculates the convex hull of the contour points.
    Extend: Ratio of the area and the area of the bounding box. Expresses how
            spread out the contour is.
    Ellipse: Fit an ellipse around the contour.
    IsConvex: Boolean value specifying if the set of contour points is convex.
    Length: Length of the contour
    Moments: Dictionary of moments.
    Perimiter: Permiter of the contour - equivalent to the length.
    RotatedBox: Rotated rectangle as a Box2D structure.

    Returns: Dictionary with key equal to the property name.

    Example: 
        image, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST,
                                                      cv2.CHAIN_APPROX_SIMPLE)
        goodContours = []
        for contour in contours:
            vals = IAMLTools.getContourProperties(contour, ["Area", "Length",
                                                            "Centroid", "Extend",
                                                            "ConvexHull"])
            if vals["Area"] > 100 and vals["Area"] < 200:
                goodContours.append(contour)
    """
    return BlobProperties.Instance.getContourProperties(contour, properties)
