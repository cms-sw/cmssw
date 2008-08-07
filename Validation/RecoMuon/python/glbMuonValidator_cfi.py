import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.RecoMuonValidatorCommon_cfi import *
glbMuonValidator = cms.EDFilter("RecoMuonValidator",
    RecoMuonValidatorCommon,
    MuonServiceProxy,
    recoLabel = cms.InputTag("globalMuons"),
    outputFileName = cms.untracked.string('validationPlots.root'),
    doAssoc = cms.untracked.bool(False),
    assocLabel = cms.InputTag("tpToGlbTrackAssociation"),
    subDir = cms.untracked.string('RecoMuonV/MuonValidator/'),
    simLabel = cms.InputTag("mergedtruth","MergedTrackTruth")
)


