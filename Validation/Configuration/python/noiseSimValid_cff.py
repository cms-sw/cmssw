import FWCore.ParameterSet.Config as cms

from Validation.TrackerHits.trackerHitsValidation_cff import *
from Validation.TrackerDigis.trackerDigisValidation_cff import *
from Validation.TrackerRecHits.trackerRecHitsValidation_cff import *
from Validation.EcalHits.ecalSimHitsValidationSequence_cff import *
from Validation.EcalDigis.ecalDigisValidationSequence_cff import *
from Validation.EcalRecHits.ecalRecHitsValidationSequence_cff import *
from Validation.HcalHits.HcalSimHitStudy_cfi import *
from Validation.HcalDigis.hcalDigisValidationSequence_cff import *
from Validation.MuonHits.muonHitsValidation_cfi import *
from Validation.MuonDTDigis.dtDigiValidation_cfi import *
from Validation.MuonCSCDigis.cscDigiValidation_cfi import *
from Validation.MuonRPCDigis.validationMuonRPCDigis_cfi import *

noiseSimValid = cms.Sequence(trackerHitsValidation+trackerDigisValidation+trackerRecHitsValidation+ecalSimHitsValidationSequence+ecalDigisValidationSequence+ecalRecHitsValidationSequence+hcalSimHitStudy+hcalDigisValidationSequence+validSimHit+muondtdigianalyzer+cscDigiValidation+validationMuonRPCDigis)


