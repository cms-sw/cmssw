import FWCore.ParameterSet.Config as cms

# Digitization of Ecal and Hcal
from SimCalorimetry.Configuration.ecalDigiSequence_cff import *
from SimCalorimetry.Configuration.hcalDigiSequence_cff import *
calDigi = cms.Sequence(ecalDigiSequence+hcalDigiSequence)

