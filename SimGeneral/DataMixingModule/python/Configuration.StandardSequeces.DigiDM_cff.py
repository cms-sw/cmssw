import FWCore.ParameterSet.Config as cms

# "Clean up" digitization to make trigger primitives
# from the new "mixed" calo cells
# and to zero-suppress them for further processing.
#
# Run after the DataMixer only.
#
# Calorimetry Digis (Ecal + Hcal) - * unsuppressed *
# returns sequence "calDigi"
#
from SimCalorimetry.Configuration.SimCalorimetryDM_cff import *
#
#
from SimGeneral.Configuration.SimGeneral_cff import *
doAllDigi = cms.Sequence(calDigi)
pdigi = cms.Sequence(doAllDigi)


