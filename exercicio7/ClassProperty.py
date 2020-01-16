#<!--------------------------------------------------------------------------->
#<!--                   ITU - IT University of Copenhage                    -->
#<!--                      Computer Science Department                      -->
#<!--                    Eye Information Research Group                     -->
#<!--       Introduction to Image Analysis and Machine Learning Course      -->
#<!-- File       : ClassProperty.py                                         -->
#<!-- Description: Class for managing the direct access to non-instanced    -->
#<!--              objects in IAMLTools                                     -->
#<!-- Author     : Fabricio Batista Narcizo                                 -->
#<!--            : Rued Langgaards Vej 7 - 4D25 - DK-2300 - Kobenhavn S.    -->
#<!--            : fabn[at]itu[dot]dk                                       -->
#<!-- Responsable: Dan Witzner Hansen (witzner[at]itu[dot]dk)               -->
#<!--              Fabricio Batista Narcizo (fabn[at]itu[dot]dk)            -->
#<!-- Information: This class is based on an example available in Stack     -->
#<!--              Overflow Website (http://goo.gl/5YUJAQ)                  -->
#<!-- Date       : 12/03/2018                                               -->
#<!-- Change     : 12/03/2018 - Creation of this class                      -->
#<!-- Review     : 12/03/2018 - Finalized                                   -->
#<!--------------------------------------------------------------------------->

__version__ = "$Revision: 2018031201 $"

########################################################################
class ClassProperty(object):

    #----------------------------------------------------------------------#
    #                   ClassProperty Class Constructor                    #
    #----------------------------------------------------------------------#
    """Class for managing the direct access to non-instanced objects."""
    def __init__(self, getter, instance="0"):
        """ClassProperty Class Constructor."""
        self.getter   = getter
        self.instance = instance

    #----------------------------------------------------------------------#
    #                            Class Methods                             #
    #----------------------------------------------------------------------#
    def __get__(self, instance, owner):
        """Get the current object instance."""
        return self.getter(owner)
