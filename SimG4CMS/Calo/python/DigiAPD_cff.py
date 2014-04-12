import FWCore.ParameterSet.Config as cms

# Calorimetry Digis (Ecal + Hcal) - * unsuppressed *
# returns sequence "calDigi"
from SimCalorimetry.Configuration.ecalDigiSequence_cff import *
from SimGeneral.Configuration.SimGeneral_cff import *

doAllDigi = cms.Sequence(ecalDigiSequence)
pdigi = cms.Sequence(cms.SequencePlaceholder("randomEngineStateProducer")*cms.SequencePlaceholder("mix")*doAllDigi)
