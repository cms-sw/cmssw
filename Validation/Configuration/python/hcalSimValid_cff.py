import FWCore.ParameterSet.Config as cms

# HCAL validation sequences
#
from Validation.HcalHits.HcalSimHitStudy_cfi import *
from Validation.HcalDigis.hcalDigisValidationSequence_cff import *
from Validation.HcalRecHits.hcalRecHitsValidationSequence_cff import *
from Validation.CaloTowers.calotowersValidationSequence_cff import *
hcalSimValid = cms.Sequence(hcalSimHitStudy+hcalDigisValidationSequence+hcalRecHitsValidationSequence+calotowersValidationSequence)


