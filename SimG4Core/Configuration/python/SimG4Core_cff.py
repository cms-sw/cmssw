import FWCore.ParameterSet.Config as cms

# Geometry and Magnetic field must be initialized separately
# Geant4-based CMS Detector simulation (OscarProducer)
# - returns label "g4SimHits"
#
from SimG4Core.Application.g4SimHits_cfi import *

from Configuration.StandardSequences.Eras import eras
eras.phase2_hcal.toModify( g4SimHits, HCalSD = dict( TestNumberingScheme = True ) )
eras.phase2_timing.toModify( g4SimHits.ECalSD, 
                             StoreLayerTimeSim = cms.untracked.bool(True),
                             TimeSliceUnit = cms.double(0.001) )
