import FWCore.ParameterSet.Config as cms
from RecoMuon.TrackingTools.MuonServiceProxy_cff import MuonServiceProxy
from Validation.RecoMuon.selectors_cff import muonTPSet

recoMuonValidator = cms.EDAnalyzer("RecoMuonValidator",
    MuonServiceProxy,
    tpSelector = muonTPSet,

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
    subDir = cms.untracked.string('Muons/RecoMuonV/'),

    #
    # Histogram dimensions     #
    #
    nBinP = cms.untracked.uint32(25),
    minP = cms.untracked.double(0.0),
    maxP = cms.untracked.double(200.0),

    nBinPt = cms.untracked.uint32(25),
    minPt = cms.untracked.double(0.0),
    maxPt = cms.untracked.double(200.0),

    doAbsEta = cms.untracked.bool(False),
    nBinEta = cms.untracked.uint32(50),
    minEta = cms.untracked.double(-2.5),
    maxEta = cms.untracked.double(2.5),

    nBinPhi = cms.untracked.uint32(25),

    # Pull width     #
    nBinPull = cms.untracked.uint32(25),
    wPull = cms.untracked.double(10.0),

    nBinErr = cms.untracked.uint32(25),

    # |p| resolution     #
    minErrP = cms.untracked.double(-0.2),
    maxErrP = cms.untracked.double(0.2),

    # pT resolution     #
    minErrPt = cms.untracked.double(-0.2),
    maxErrPt = cms.untracked.double(0.2),

    # q/pT resolution     #
    minErrQPt = cms.untracked.double(-5.0),
    maxErrQPt = cms.untracked.double(5.0),

    # Eta resolution     #
    minErrEta = cms.untracked.double(-0.02),
    maxErrEta = cms.untracked.double(0.02),

    # Phi resolution     #
    minErrPhi = cms.untracked.double(-0.1),
    maxErrPhi = cms.untracked.double(0.1),

    # Dxy resolution     #
    minErrDxy = cms.untracked.double(-0.1),
    maxErrDxy = cms.untracked.double(0.1),

    # Dz resolution     #
    minErrDz = cms.untracked.double(-0.1),
    maxErrDz = cms.untracked.double(0.1),

    # Number of sim-reco associations     #
    nAssoc = cms.untracked.uint32(10),

    # Number of sim,reco Tracks     #
    nTrks = cms.untracked.uint32(50)
)

