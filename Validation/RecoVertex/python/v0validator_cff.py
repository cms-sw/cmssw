import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.MagneticField_cff import *
from DQMServices.Core.DQM_cfg import *
DQMStore.collateHistograms = cms.untracked.bool(True)
from Validation.RecoVertex.v0validator_cfi import *
