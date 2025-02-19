import FWCore.ParameterSet.Config as cms

# BTagPerformanceAnalyzer configuration
from Validation.RecoB.bTagAnalysis_cfi import *

bTagValidationHarvest = bTagValidation.clone()

bTagValidationHarvest.finalizePlots = True
bTagValidationHarvest.finalizeOnly = True
#
