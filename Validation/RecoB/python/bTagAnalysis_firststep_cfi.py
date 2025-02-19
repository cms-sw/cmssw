import FWCore.ParameterSet.Config as cms

# BTagPerformanceAnalyzer configuration
from Validation.RecoB.bTagAnalysis_cfi import *

bTagValidationFirstStep = bTagValidation.clone()

bTagValidationFirstStep.finalizePlots = False
bTagValidationFirstStep.finalizeOnly = False
#
