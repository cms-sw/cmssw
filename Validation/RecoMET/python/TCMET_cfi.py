import FWCore.ParameterSet.Config as cms

# File: TCMET.cfi
# Author: R. Remington
# Date: 11.14.2008
#
# Fill validation histograms for MET.

tcMetAnalyzer = cms.EDAnalyzer(
    "METTester",
    InputMETLabel = cms.InputTag("tcMet"),
    InputCaloMETLabel = cms.InputTag("met"),
    InputTrackLabel = cms.InputTag("generalTracks"),
    InputMuonLabel = cms.InputTag("muons"),
    InputElectronLabel = cms.InputTag("gsfElectrons"),
    InputBeamSpotLabel = cms.InputTag("offlineBeamSpot"),
    METType = cms.untracked.string('TCMET'),
    FineBinning = cms.untracked.bool(True),
    FolderName = cms.untracked.string("RecoMETV/MET_Global/"),
    minhits = cms.int32(6),
    maxd0 = cms.double(0.1),
    maxchi2 = cms.double(5),
    maxeta = cms.double(2.65),
    maxpt = cms.double(100.),
    maxPtErr = cms.double(0.2),
    trkQuality = cms.vint32(2),
    trkAlgos = cms.vint32()
    ) 


