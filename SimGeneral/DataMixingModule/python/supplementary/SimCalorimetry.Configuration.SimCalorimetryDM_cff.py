import FWCore.ParameterSet.Config as cms

from SimCalorimetry.Configuration.ecalDigiSequenceDM_cff import *
from SimCalorimetry.Configuration.hcalDigiSequenceDM_cff import *
from SimCalorimetry.Configuration.castorDigiSequenceDM_cff import *
#calDigi = cms.Sequence(ecalDigiSequence+hcalDigiSequence+castorDigiSequence)
calDigi = cms.Sequence(ecalDigiSequence+hcalDigiSequence)

