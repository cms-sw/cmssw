import FWCore.ParameterSet.Config as cms

from SimCalorimetry.Configuration.ecalDigiSequence_cff import *
from SimCalorimetry.Configuration.hcalDigiSequence_cff import *
from SimCalorimetry.Configuration.castorDigiSequence_cff import *
calDigi = cms.Sequence(ecalDigiSequence+hcalDigiSequence+castorDigiSequence)

# fastsim has no castor model
from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    calDigi.remove(castorDigiSequence)
