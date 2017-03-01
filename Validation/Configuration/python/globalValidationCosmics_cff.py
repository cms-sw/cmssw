import FWCore.ParameterSet.Config as cms

from SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi import *
from Validation.RecoMuon.muonValidation_cff import *

# filter/producer "pre-" sequence for globalValidation
globalPrevalidationCosmics = cms.Sequence(simHitTPAssocProducer)

globalValidationCosmics = cms.Sequence(recoCosmicMuonValidation)
