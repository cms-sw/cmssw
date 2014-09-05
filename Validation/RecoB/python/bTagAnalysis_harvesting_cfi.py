import FWCore.ParameterSet.Config as cms

# BTagPerformanceAnalyzer configuration
from Validation.RecoB.bTagAnalysis_cfi import *
bTagValidationHarvest = bTagHarvestMC.clone()

from DQMOffline.RecoB.bTagAnalysisData_cfi import *
bTagValidationHarvestData = bTagHarvest.clone()
