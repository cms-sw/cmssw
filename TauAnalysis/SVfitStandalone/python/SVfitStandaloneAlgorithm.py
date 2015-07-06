from TauAnalysis.SVfitStandalone import loadlibs

from ROOT import SVfitStandaloneAlgorithm

# importing the python binding to the C++ class from ROOT 

class SVfitAlgo( SVfitStandaloneAlgorithm ):
    '''Just an additional wrapper, not really needed :-)
    We just want to illustrate the fact that you could
    use such a wrapper to add functions, attributes, etc,
    in an improved interface to the original C++ class. 
    '''
    def __init__ (self, *args) :
        super(SVfitAlgo, self).__init__(*args) 
