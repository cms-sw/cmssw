import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQM_cfg import *
DQMStore.collateHistograms = cms.untracked.bool(True)
from Validation.HGCalValidation.HGCalValidator_cfi import *

