import FWCore.ParameterSet.Config as cms

# File: CaloMET.cfi
# Author: B. Scurlock & R. Remington
# Date: 03.04.2008
#
# Fill validation histograms for MET. Assumes met, metNoHF, metOpt, metOptNoHF are in the event
from Validation.RecoMET.CaloMET_cfi import *
analyzeCaloMETNoHO = cms.Sequence(metAnalyzer*metNoHFAnalyzer*metOptAnalyzer*metOptNoHFAnalyzer)
analyzeCaloMETHO = cms.Sequence(metHOAnalyzer*metNoHFHOAnalyzer*metOptHOAnalyzer*metOptNoHFHOAnalyzer)
analyzeCaloMET = cms.Sequence(analyzeCaloMETNoHO*analyzeCaloMETHO)
