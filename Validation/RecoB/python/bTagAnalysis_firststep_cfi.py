import FWCore.ParameterSet.Config as cms

# BTagPerformanceAnalyzer configuration
from Validation.RecoB.bTagAnalysis_cfi import *

bTagValidationFirstStep = bTagValidation.clone()
bTagValidationFirstStep.finalizePlots = False
bTagValidationFirstStep.finalizeOnly = False


from DQMOffline.RecoB.bTagAnalysisData_cfi import *

bTagValidationFirstStepData = bTagAnalysis.clone()
bTagValidationFirstStepData.finalizePlots = False
bTagValidationFirstStepData.finalizeOnly = False

