import FWCore.ParameterSet.Config as cms

# BTagPerformanceAnalyzer configuration
from Validation.RecoB.bTagAnalysis_cfi import *
bTagValidationFirstStep = bTagValidation.clone()

from DQMOffline.RecoB.bTagAnalysisData_cfi import *
bTagValidationFirstStepData = bTagAnalysis.clone()

# foo bar baz
# h9gOB5YLPPJda
# VeldZ2ltWFm7C
