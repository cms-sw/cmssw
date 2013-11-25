import FWCore.ParameterSet.Config as cms

from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching

SimpleMuon = cms.EDAnalyzer('SimpleMuon',
   strips = cms.PSet()                 
)    
