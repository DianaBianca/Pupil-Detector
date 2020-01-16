#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhage                    -->
#<!--                      Computer Science Department                      -->
#<!--                    Eye Information Research Group                     -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : BlobProperties.py                                        -->
#<!-- Description: Get descriptors of contour-based connected components    -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: You DO NOT need to change this file                      -->
#<!-- Date       : 12/03/2018                                               -->
#<!-- Change     : 12/03/2018 - Creation of this class                      -->
#<!-- Review     : 12/03/2018 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2018031201 $"

########################################################################
import cv2
import math

from ClassProperty import ClassProperty

########################################################################
class BlobProperties:
    """BlobProperties Class is used to manager the blob properties extraction."""

    #----------------------------------------------------------------------#
    #                           Class Attributes                           #
    #----------------------------------------------------------------------#
    __Instance = None

    #----------------------------------------------------------------------#
    #                         Static Class Methods                         #
    #----------------------------------------------------------------------#
    @ClassProperty
    def Instance(self):
        """Create an instance to manager the blob properties extraction."""
        if self.__Instance is None:
            self.__Instance = Properties()
        return self.__Instance

    #----------------------------------------------------------------------#
    #                   BlobProperties Class Constructor                   #
    #----------------------------------------------------------------------#
    def __init__(self):
        """This constructor is never used by the system."""
        pass

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __repr__(self):
        """Gets an object representation in a string format."""
        return "IAMLTools.BlobProperties object."

class Properties(object):
    """
    This class is used for getting descriptors of contour-based connected
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
    Perimeter: Permiter of the contour - equivalent to the length.
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

    #----------------------------------------------------------------------#
    #                     Properties Class Constructor                     #
    #----------------------------------------------------------------------#
    def __init__(self):
        """Properties Class Constructor."""
        pass

    #----------------------------------------------------------------------#
    #                         Public Class Methods                         #
    #----------------------------------------------------------------------#
    def getContourProperties(self, contour, properties=[]):
        """Calcule and return a list of strings specifying by properties."""
        # Initial variables.
        failInInput  = False
        props = {}

        for prop in properties:
            prop = str(prop).lower()

            if prop == "approximation":
                props.update({"Approximation" : self.__CalculateApproximation(contour)})
            if prop == "area":
                props.update({"Area" : self.__CalculateArea(contour)})
            elif prop == "boundingbox":
                props.update({"BoundingBox" : self.__CalculateBoundingBox(contour)})
            elif prop == "centroid":
                props.update({"Centroid" : self.__CalculateCentroid(contour)})
            elif prop == "circle":
                props.update({"Circle" : self.__CalculateCircle(contour)})
            elif prop == "convexhull":
                props.update({"ConvexHull" : self.__CalculateConvexHull(contour)})
            elif prop == "extend":
                props.update({"Extend" : self.__CalculateExtend(contour)})
            elif prop == "ellipse":
                props.update({"Ellipse" : self.__CalculateEllipse(contour)})
            elif prop == "isconvex":
                props.update({"IsConvex" : self.__IsConvex(contour)})
            elif prop == "length":
                props.update({"Length" : self.__CalculateLength(contour)})
            elif prop == "moments":
                props.update({"Moments" : self.__CalculateMoments(contour)})
            elif prop == "perimeter":
                props.update({"Perimeter" : self.__CalculatePerimeter(contour)})
            elif prop == "rotatedbox":
                props.update({"RotatedBox" : self.__CalculateRotatedBox(contour)})
            elif prop == "circularity":
                props.update({"Circularity" : self.__CalculateCircularity(contour)}) 
            elif failInInput:
                pass
            else:
                print("\t--" * 20)
                print("\t*** PROPERTY ERROR " + prop + " DOES NOT EXIST ***")
                print("\tTHIS ERROR MESSAGE WILL ONLY BE PRINTED ONCE")
                print("\--" * 20)
                failInInput = True

        return props

    #----------------------------------------------------------------------#
    #                        Private Class Methods                         #
    #----------------------------------------------------------------------#
    def __CalculateApproximation(self, contour):
        """
        Calculate the approximation of a contour shape to another shape with
        less number of vertices depending upon the precision we specify.         """
        epsilon = 0.1 * cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, epsilon, True)

    def __CalculateArea(self, contour):
        """
        Calculate the contour area by the function cv2.contourArea() or
        from moments, M["m00"].
        """
        return cv2.contourArea(contour)

    def __CalculateBoundingBox(self, contour):
        """
        Calculate the bouding rectangle. It is a straight rectangle, it
        doesn't consider the rotation of the object. So area of the bounding
        rectangle won't be minimum. It is found by the function
        cv2.boundingRect().
        """
        return cv2.boundingRect(contour)

    def __CalculateCentroid(self, contour):
        """
        Calculates the centroid of the contour. Moments up to the third
        order of a polygon or rasterized shape.
        """
        moments = cv2.moments(contour)

        centroid = (-1, -1)
        if moments["m00"] != 0:
            centroid = (int(round(moments["m10"] / moments["m00"])),
                        int(round(moments["m01"] / moments["m00"])))

        return centroid

    def __CalculateCircle(self, contour):
        """
        Calculate the circumcircle of an object using the function
        cv2.minEnclosingCircle(). It is a circle which completely covers
        the object with minimum area.        """
        return cv2.minEnclosingCircle(contour)

    def __CalculateConvexHull(self, contour):
        """
        Finds the convex hull of a point set by checking a curve for
        convexity defects and corrects it.
        """
        return cv2.convexHull(contour)

    def __CalculateEllipse(self, contour):
        """
        Fit an ellipse to an object. It returns the rotated rectangle
        in which the ellipse is inscribed.
        """
        if len(contour) > 5:
            return cv2.fitEllipse(contour)

        return cv2.minAreaRect(contour)

    def __CalculateExtend(self, contour):
        """
        Calculate the countour extend.
        """
        area = self.__CalculateArea(contour)
        boundingBox = self.__CalculateBoundingBox(contour)
        return area / (boundingBox[2] * boundingBox[3])

    def __IsConvex(self, contour):
        """
        Check if a curve is convex or not.
        """
        return cv2.isContourConvex(contour)

    def __CalculateLength(self, curve):
        """
        Calculate a contour perimeter or a curve length.
        """
        return cv2.arcLength(curve, True)    

    def __CalculateMoments(self, contour):
        """
        Calculate the contour moments to help you to calculate some features
        like center of mass of the object, area of the object etc.
        """
        return cv2.moments(contour) 

    def __CalculatePerimeter(self, curve):
        """Calculates a contour perimeter or a curve length."""
        return cv2.arcLength(curve, True)

    def __CalculateRotatedBox(self, contour):
        """
        Calculate the rotated rectangle as a Box2D structure which contains
        following detals: (center(x, y), (width, height), angle of rotation).
        """
        rectangle = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rectangle)
        return np.int0(box)
    
    def __CalculateCircularity(self,contour):
        
        perimeter = cv2.arcLength(contour, True)
        #print(perimeter)
        area = cv2.contourArea(contour)
        val = 0
        if (perimeter != 0 ):
            
            #print('\t',area)
            val = float (4 * 3.1415 * area)/(perimeter ** 2)
            
        return val
            