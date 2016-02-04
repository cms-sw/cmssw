import FWCore.ParameterSet.Config as cms

# File: PFMET.cfi
# Author: R. Remington
# Date: 11.14.2008
#
# Fill validation histograms for PFMET.

pfMetAnalyzer = cms.EDAnalyzer(
   "METTester",
    OutputFile = cms.untracked.string('output.root'),
    InputMETLabel = cms.InputTag("pfMet"),
    METType = cms.untracked.string('PFMET'),
   FineBinning = cms.untracked.bool(False),
   FolderName = cms.untracked.string("RecoMETV/MET_Global/")
   ) 


