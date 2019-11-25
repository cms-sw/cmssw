import FWCore.ParameterSet.Config as cms
from Validation.SiTrackerPhase2V.Phase2TrackerValidateDigi_cff import *

trackerphase2ValidationSource = cms.Sequence(pixDigiValid  + otDigiValid )
