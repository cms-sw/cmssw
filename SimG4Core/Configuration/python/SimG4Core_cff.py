import FWCore.ParameterSet.Config as cms

# Geometry and Magnetic field must be initialized separately
# Geant4-based CMS Detector simulation (OscarProducer)
# - returns label "g4SimHits"
#
from SimG4Core.Application.g4SimHits_cfi import *

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify( g4SimHits, HCalSD = dict( TestNumberingScheme = True ) )
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify( g4SimHits.ECalSD, 
                             StoreLayerTimeSim = cms.untracked.bool(True),
                             TimeSliceUnit = cms.double(0.001) )
