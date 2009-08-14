import FWCore.ParameterSet.Config as cms

# File: TCMET.cfi
# Author: R. Remington
# Date: 11.14.2008
#
# Fill validation histograms for MET.

corMetGlobalMuonsAnalyzer = cms.EDAnalyzer(
    "METTester",
    InputMETLabel = cms.InputTag("corMetGlobalMuons"),
    METType = cms.untracked.string('MuonCorrectedCaloMET'),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/")
    ) 


