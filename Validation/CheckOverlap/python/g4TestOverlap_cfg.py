import FWCore.ParameterSet.Config as cms

#------------------------------------------------------------
# This is a test of overlaps which use only Geant4 tool. 
# To start it names of Physical volumes should be provides.
# It is possible to check overlap check parameters. 
# Static build of Geant4 may be used
#------------------------------------------------------------

from Validation.CheckOverlap.testOverlap_cff import *

process.g4SimHits.CheckOverlap = True
process.g4SimHits.G4CheckOverlap = cms.PSet(
    NodeNames = cms.vstring('ECAL'),
    Tolerance = cms.untracked.double(0.0001), # in mm
    Resolution = cms.untracked.int32(10000),
    ErrorThreshold = cms.untracked.int32(1),
    Level = cms.untracked.int32(0),
    Depth = cms.untracked.int32(-1),
    Verbose = cms.untracked.bool(True)
)

