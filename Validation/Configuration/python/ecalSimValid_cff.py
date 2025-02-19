import FWCore.ParameterSet.Config as cms

# ECAL validation sequences
#
from Validation.EcalHits.ecalSimHitsValidationSequence_cff import *
from Validation.EcalDigis.ecalDigisValidationSequence_cff import *
from Validation.EcalRecHits.ecalRecHitsValidationSequence_cff import *
from Validation.EcalClusters.ecalClustersValidationSequence_cff import *
ecalSimValid = cms.Sequence(ecalSimHitsValidationSequence+ecalDigisValidationSequence+ecalRecHitsValidationSequence+ecalClustersValidationSequence)


