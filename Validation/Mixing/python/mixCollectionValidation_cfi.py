import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.mixObjects_cfi import *

mixCollectionValidation = cms.EDFilter("MixCollectionValidation",
    outputFile = cms.string('histosMixCollStandardMM.root'),
    minBunch = cms.int32(-5),
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
