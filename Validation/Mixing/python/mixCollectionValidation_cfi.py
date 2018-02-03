import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.mixObjects_cfi import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
mixCollectionValidation = DQMEDAnalyzer('MixCollectionValidation',
#    outputFile = cms.string('histosMixCollStandardMM.root'),
    outputFile = cms.string(''),
    minBunch = cms.int32(-12),
    maxBunch = cms.int32(3),
    verbose = cms.untracked.bool(False),                                       
    mixObjects = cms.PSet(
        mixCH = cms.PSet(                   
            mixCaloHits                     
        ),                      
        mixTracks = cms.PSet(       
            mixSimTracks                    
        ),                      
        mixVertices = cms.PSet(     
            mixSimVertices                  
        ),                      
        mixSH = cms.PSet(           
            mixSimHits                      
        ),                      
        mixHepMC = cms.PSet(        
            mixHepMCProducts                
        )                       
    )                               
)                                   
