import FWCore.ParameterSet.Config as cms

# Geometry and Magnetic field must be initialized separately
# Geant4-based CMS Detector simulation (OscarProducer)
# - returns label "g4SimHits"
#
from SimG4Core.Application.g4SimHits_cfi import *

def _modifySimPhase2Common( theProcess ):
    theProcess.g4SimHits.HCalSD.TestNumberingScheme = True

from Configuration.StandardSequences.Eras import eras
modifySimPhase2Common_ = eras.phase2_common.makeProcessModifier( _modifySimPhase2Common )
