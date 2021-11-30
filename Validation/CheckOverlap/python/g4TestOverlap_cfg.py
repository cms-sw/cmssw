import FWCore.ParameterSet.Config as cms

#------------------------------------------------------------
# This is a test of overlaps which use only Geant4 tool. 
# To start it names of Physical volumes should be provides.
# It is possible to check overlap check parameters. 
# Static build of Geant4 may be used
#------------------------------------------------------------

from Validation.CheckOverlap.testOverlap_cff import *

process.g4SimHits.CheckOverlap = True
process.g4SimHits.G4CheckOverlap = dict(
    NodeNames = 'ECAL',
    Tolerance = 0.0001, # in mm
    Resolution = 10000,
    ErrorThreshold = 1,
    Level = 0,
    Depth = -1,
    Verbose = True
)

