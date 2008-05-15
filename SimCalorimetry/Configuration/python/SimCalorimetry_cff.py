import FWCore.ParameterSet.Config as cms

from SimCalorimetry.Configuration.ecalDigiSequence_cff import *
from SimCalorimetry.Configuration.hcalDigiSequence_cff import *
calDigi = cms.Sequence(ecalDigiSequence+hcalDigiSequence)

