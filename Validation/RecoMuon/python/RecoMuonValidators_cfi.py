import FWCore.ParameterSet.Config as cms

RecoMuonHistoDefaults = cms.PSet(
    nBinEta = cms.untracked.uint32(50),
    nBinPull = cms.untracked.uint32(50),
    nBinErrPt = cms.untracked.uint32(50),
    minPt = cms.untracked.double(0.0),
    wErrPt = cms.untracked.double(50.0),
    nBinPt = cms.untracked.uint32(50),
    maxEta = cms.untracked.double(2.4),
    minEta = cms.untracked.double(0.0),
    doAbsEta = cms.untracked.bool(True),
    wErrQOverPt = cms.untracked.double(3.0),
    nBinErrQOverPt = cms.untracked.uint32(50),
    nBinPhi = cms.untracked.uint32(50),
    maxPt = cms.untracked.double(50.0),
    wPull = cms.untracked.double(5.0)
)
recoMuonValidation = cms.EDFilter("RecoMuonValidator",
    MuonServiceProxy,
    GlbMuonHistoParameters = cms.PSet(
        RecoMuonHistoDefaults,
        subDir = cms.untracked.string('GlbDeltaR')
    ),
    SimPtcl = cms.InputTag("trackingParticles"),
    outputFileName = cms.untracked.string('RecoMuonValidation.root'),
    RecoMuon = cms.InputTag("muons"),
    histoManager = cms.untracked.string('DQM'),
    StaMuonHistoParameters = cms.PSet(
        RecoMuonHistoDefaults,
        subDir = cms.untracked.string('StaDeltaR')
    )
)


