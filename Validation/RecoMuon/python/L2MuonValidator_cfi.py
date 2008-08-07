import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.RecoMuonValidatorCommon_cfi import *
L2MuonValidator = cms.EDFilter("RecoMuonValidator",
    RecoMuonValidatorCommon,
    MuonServiceProxy,
    recoLabel = cms.InputTag("hltL2Muons","UpdatedAtVtx"),
    outputFileName = cms.untracked.string('validationPlots.root'),
    doAssoc = cms.untracked.bool(False),
    assocLabel = cms.InputTag("tpToL2TrackAssociation"),
    subDir = cms.untracked.string('RecoMuonV/MuonValidator/'),
    simLabel = cms.InputTag("mergedtruth","MergedTrackTruth")
)


