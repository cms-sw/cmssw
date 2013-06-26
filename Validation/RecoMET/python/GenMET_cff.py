import FWCore.ParameterSet.Config as cms

# File: GenMET.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for MET. Assumes genMet and genMetNoNuBSM are in the event.
from Validation.RecoMET.GenMET_cfi import *
analyzeGenMET = cms.Sequence(genMetTrueAnalyzer*genMetCaloAnalyzer*genMetCaloAndNonPromptAnalyzer)
