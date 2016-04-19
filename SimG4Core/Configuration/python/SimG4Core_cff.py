import FWCore.ParameterSet.Config as cms

# Geometry and Magnetic field must be initialized separately
# Geant4-based CMS Detector simulation (OscarProducer)
# - returns label "g4SimHits"
#
from SimG4Core.Application.g4SimHits_cfi import *

def _modifySimPhase2Common( obj ):
    obj.HCalSD.TestNumberingScheme = True

from Configuration.StandardSequences.Eras import eras
eras.phase2_common.toModify( g4SimHits, func=_modifySimPhase2Common )
