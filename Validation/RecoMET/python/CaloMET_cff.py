import FWCore.ParameterSet.Config as cms

# File: CaloMET.cfi
# Author: B. Scurlock & R. Remington
# Date: 03.04.2008
#
# Fill validation histograms for MET. Assumes met, metNoHF, metOpt, metOptNoHF are in the event
# Added met only sequence. Removing the unsed met collections to reduce the load
# on DQM server. - Samantha Hewamanage (samantha@cern.ch) 04-26-2012

from Validation.RecoMET.CaloMET_cfi import *
analyzeCaloMETNoHO = cms.Sequence(metAnalyzer*metNoHFAnalyzer*metOptAnalyzer*metOptNoHFAnalyzer)
analyzeCaloMETHO = cms.Sequence(metHOAnalyzer*metNoHFHOAnalyzer*metOptHOAnalyzer*metOptNoHFHOAnalyzer)
analyzeCaloMET = cms.Sequence(analyzeCaloMETNoHO*analyzeCaloMETHO)
analyzeCaloMETonly = cms.Sequence(metAnalyzer)
