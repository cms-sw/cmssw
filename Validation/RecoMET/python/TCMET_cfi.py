import FWCore.ParameterSet.Config as cms

# File: TCMET.cfi
# Author: R. Remington
# Date: 11.14.2008
#
# Fill validation histograms for MET.

tcMetAnalyzer = cms.EDAnalyzer(
    "METTester",
    InputMETLabel = cms.InputTag("tcMet"),
    METType = cms.untracked.string('TCMET'),
    FineBinning = cms.untracked.bool(True)
    ) 


