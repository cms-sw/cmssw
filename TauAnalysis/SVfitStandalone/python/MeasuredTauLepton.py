from TauAnalysis.SVfitStandalone import loadlibs

from ROOT import svFitStandalone

# importing the python binding to the C++ class from ROOT 

class measuredTauLepton( svFitStandalone.MeasuredTauLepton ):
    '''
       decayType : {
                    0:kUndefinedDecayType,
                    1:kTauToHadDecay, 
                    2:kTauToElecDecay,
                    3:kTauToMuDecay,  
                    4:kPrompt
                   }          
    '''
    
    def __init__(self, decayType, pt, eta, phi, mass, decayMode=-1):        
        super(measuredTauLepton, self).__init__(decayType, pt, eta, phi, mass, decayMode) 
