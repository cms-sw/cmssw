import FWCore.ParameterSet.Config as cms
from RecoMuon.TrackingTools.MuonServiceProxy_cff import MuonServiceProxy

recoMuonValidator = cms.EDFilter("RecoMuonValidator",
    MuonServiceProxy,

    simLabel = cms.InputTag("mergedtruth","MergedTrackTruth"),
    trkMuLabel = cms.InputTag("generalTracks"),
    staMuLabel = cms.InputTag("standalonemuons:UpdatedAtVtx"),
    glbMuLabel = cms.InputTag("globalMuons"),
    muonLabel = cms.InputTag("muons"),

    trkMuAssocLabel = cms.InputTag("TrackAssociatorByHits"),
    staMuAssocLabel = cms.InputTag("TrackAssociatorByHits"),
    glbMuAssocLabel = cms.InputTag("TrackAssociatorByHits"),
    doAssoc = cms.untracked.bool(False),

    outputFileName = cms.untracked.string(''),
    subDir = cms.untracked.string('RecoMuonV/'),

    #
    # Histogram dimensions     #
    #
    nBinP = cms.untracked.uint32(50),
    minP = cms.untracked.double(0.0),
    maxP = cms.untracked.double(1000.0),

    nBinPt = cms.untracked.uint32(50),
    minPt = cms.untracked.double(0.0),
    maxPt = cms.untracked.double(1000.0),

    doAbsEta = cms.untracked.bool(True),
    nBinEta = cms.untracked.uint32(50),
    minEta = cms.untracked.double(0.0),
    maxEta = cms.untracked.double(2.4),

    nBinPhi = cms.untracked.uint32(50),

    # Pull width     #
    nBinPull = cms.untracked.uint32(50),
    wPull = cms.untracked.double(10.0),

    nBinErr = cms.untracked.uint32(50),

    # |p| resolution     #
    minErrP = cms.untracked.double(-0.1),
    maxErrP = cms.untracked.double(0.1),

    # pT resolution     #
    minErrPt = cms.untracked.double(-0.1),
    maxErrPt = cms.untracked.double(0.1),

    # q/pT resolution     #
    minErrQPt = cms.untracked.double(-10.0),
    maxErrQPt = cms.untracked.double(10.0),

    # Eta resolution     #
    minErrEta = cms.untracked.double(-0.01),
    maxErrEta = cms.untracked.double(0.01),

    # Phi resolution     #
    minErrPhi = cms.untracked.double(-0.003),
    maxErrPhi = cms.untracked.double(0.003),

    # Dxy resolution     #
    minErrDxy = cms.untracked.double(-0.01),
    maxErrDxy = cms.untracked.double(0.01),

    # Dz resolution     #
    minErrDz = cms.untracked.double(-0.01),
    maxErrDz = cms.untracked.double(0.01),

    # Number of sim-reco associations     #
    nAssoc = cms.untracked.uint32(50),

    # Number of sim,reco Tracks     #
    nTrks = cms.untracked.uint32(50),

    # Number of hits     #
    nHits = cms.untracked.uint32(70)
)

