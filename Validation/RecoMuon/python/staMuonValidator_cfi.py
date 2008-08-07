import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.RecoMuonValidatorCommon_cfi import *
staMuonValidator = cms.EDFilter("RecoMuonValidator",
    RecoMuonValidatorCommon,
    MuonServiceProxy,
    recoLabel = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
    outputFileName = cms.untracked.string('validationPlots.root'),
    doAssoc = cms.untracked.bool(False),
    assocLabel = cms.InputTag("tpToStaTrackAssociation"),
    subDir = cms.untracked.string('RecoMuonV/MuonValidator/'),
    simLabel = cms.InputTag("mergedtruth","MergedTrackTruth")
)


