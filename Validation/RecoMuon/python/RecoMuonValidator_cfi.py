import FWCore.ParameterSet.Config as cms
from RecoMuon.TrackingTools.MuonServiceProxy_cff import MuonServiceProxy
from Validation.RecoMuon.selectors_cff import muonTPSet

recoMuonValidator = cms.EDAnalyzer("RecoMuonValidator",
    MuonServiceProxy,
    tpSelector = muonTPSet,

    usePFMuon = cms.untracked.bool(False),

    simLabel = cms.InputTag("mix","MergedTrackTruth"),
    muonLabel = cms.InputTag("muons"),

    muAssocLabel = cms.InputTag("MuonAssociatorByHits"),

    doAssoc = cms.untracked.bool(True),

    outputFileName = cms.untracked.string(''),
    subDir = cms.untracked.string('Muons/RecoMuonV/'),
    trackType = cms.string("global"),
    #string cut selection
    selection = cms.string("isTrackerMuon && muonID('TMLastStationAngTight')"),

    wantTightMuon = cms.bool(False),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    primaryVertex = cms.InputTag('offlinePrimaryVertices'),

    #
    # Histogram dimensions     #
    #
                                   # reduce to 60 from 100, bins or 25#
    nBinP = cms.untracked.uint32(60), 
    minP = cms.untracked.double(0.0),
    maxP = cms.untracked.double(1500.0),

    nBinPt = cms.untracked.uint32(60),
    minPt = cms.untracked.double(0.0),
    maxPt = cms.untracked.double(1500.0),

    doAbsEta = cms.untracked.bool(False),
                                   
                                   # reduce to 25 from 100, bins aprox 0.2\\#
    nBinEta = cms.untracked.uint32(25),  
    minEta = cms.untracked.double(-2.5),
    maxEta = cms.untracked.double(2.5),
                                   
                                   # reduce to 50 from 100, bins of 0.06#
    nBinDxy = cms.untracked.uint32(60), 
    minDxy = cms.untracked.double(-1.5),
    maxDxy = cms.untracked.double(1.5),

                                   # reduce to 50, from 100 bin aprox 1.#
    nBinDz = cms.untracked.uint32(50), 
    minDz = cms.untracked.double(-25.),
    maxDz = cms.untracked.double(25.),

                                   # reduce to 15, from 25#
    nBinPhi = cms.untracked.uint32(15), 

    # Pull width     #
                                   # reduce to 25 from 50, bins aprox 0.2#
    nBinPull = cms.untracked.uint32(25), 
    wPull = cms.untracked.double(5.0),

    nBinErr = cms.untracked.uint32(30), # reduce from 50, for all#

    # |p| resolution     #
    minErrP = cms.untracked.double(-0.3),
    maxErrP = cms.untracked.double(0.3),

    # pT resolution     #
    minErrPt = cms.untracked.double(-0.3),
    maxErrPt = cms.untracked.double(0.3),

    # q/pT resolution     #
    minErrQPt = cms.untracked.double(-0.1),
    maxErrQPt = cms.untracked.double(0.1),

    # Eta resolution     #
    minErrEta = cms.untracked.double(-0.01),
    maxErrEta = cms.untracked.double(0.01),

    # Phi resolution     #
    minErrPhi = cms.untracked.double(-0.05),
    maxErrPhi = cms.untracked.double(0.05),

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
