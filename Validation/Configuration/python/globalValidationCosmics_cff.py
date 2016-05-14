import FWCore.ParameterSet.Config as cms

from SimGeneral.TrackingAnalysis.simHitTPAssociation_cfi import *
from Validation.RecoMuon.muonValidation_cff import *
# ADD new validation
from Validation.RecoMuon.NewMuonValidation_cff import *

# filter/producer "pre-" sequence for globalValidation
globalPrevalidationCosmics = cms.Sequence(simHitTPAssocProducer)

# to be customized for OLD or NEW validation
#globalValidationCosmics = cms.Sequence(recoCosmicMuonValidation)
globalValidationCosmics = cms.Sequence(NEWrecoCosmicMuonValidation)
