import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
#service = DaqMonitorROOTBackEnd{}
ValidationHistoParameters = cms.PSet(
    nBinEta = cms.untracked.uint32(98), ## number of EtaBin (= 49 in PTDR plots)

    nBinPull = cms.untracked.uint32(50),
    minPt = cms.untracked.double(0.0), ## minimum pT

    maxEta = cms.untracked.double(2.4), ## maximum eta

    minEta = cms.untracked.double(-2.4), ## minimum eta

    nBinErrQPt = cms.untracked.uint32(300),
    nHits = cms.untracked.uint32(70),
    widthGlbErrQPt = cms.untracked.double(2.0), ## maximum simga(q/pT) for Global muons

    widthSeedErrQPt = cms.untracked.double(3.0), ## maximum sigma(q/pT) for Seeds

    widthPull = cms.untracked.double(5.0),
    nBinPhi = cms.untracked.uint32(50), ## number of phi Bin

    maxPt = cms.untracked.double(5000.0), ## maximum pT

    widthStaErrQPt = cms.untracked.double(2.0) ## maximum simga(q/pT) for Standalone muons

)
SingleMuonValidator = cms.EDFilter("RecoMuonValidator",
    ValidationHistoParameters,
    MuonServiceProxy,
    tkMinP = cms.double(2.5),
    TkTrack = cms.InputTag("ctfWithMaterialTracks"),
    tkMinPt = cms.double(1.0),
    GlbTrack = cms.InputTag("globalMuons"),
    Seed = cms.InputTag("MuonSeed"),
    staMinRho = cms.double(1.0),
    SeedPropagator = cms.string('SteppingHelixPropagatorAny'),
    staMinPt = cms.double(1.0),
    StaTrack = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
    subDir = cms.string('Offline'),
    staMinR = cms.double(2.5),
    outputFileName = cms.untracked.string('SingleMuonValidation.root'),
    SimTrack = cms.InputTag("g4SimHits")
)



