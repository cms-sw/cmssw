import FWCore.ParameterSet.Config as cms

# File: GenMETFromGenJets.cfi
# Author: B. Scurlock
# Date: 03.04.2008
#
# Fill validation histograms for MET. Assumes genMetIC5GenJets is in the event.
from Validation.RecoMET.GenMETFromGenJets_cfi import *
analyzeGenMETFromGenJets = cms.Sequence(genMetIC5GenJetsAnalyzer)

