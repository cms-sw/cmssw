import FWCore.ParameterSet.Config as cms

# BTagPerformanceAnalyzer configuration
from Validation.RecoB.bTagAnalysis_cfi import *

bTagValidationHarvest = bTagValidation.clone()
bTagValidationHarvest.finalizePlots = True
bTagValidationHarvest.finalizeOnly = True


from DQMOffline.RecoB.bTagAnalysisData_cfi import *

bTagValidationHarvestData = bTagAnalysis.clone()
bTagValidationHarvestData.finalizePlots = True
bTagValidationHarvestData.finalizeOnly = True
