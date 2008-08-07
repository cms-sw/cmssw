import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *

RecoMuonValidatorCommon = cms.PSet(
    nBinEta = cms.untracked.uint32(50),
    minErrEta = cms.untracked.double(-0.01),
    nBinPt = cms.untracked.uint32(50),
    maxEta = cms.untracked.double(2.4),
    maxP = cms.untracked.double(1000.0),
    minErrDz = cms.untracked.double(-0.01),
    nBinErr = cms.untracked.uint32(50),
    nAssoc = cms.untracked.uint32(50),
    nBinPull = cms.untracked.uint32(50),
    minErrPhi = cms.untracked.double(-0.003),
    minErrP = cms.untracked.double(-0.1),
    doAbsEta = cms.untracked.bool(True),
    minP = cms.untracked.double(0.0),
    wPull = cms.untracked.double(10.0),
    minErrPt = cms.untracked.double(-0.1),
    minErrDxy = cms.untracked.double(-0.01),
    minPt = cms.untracked.double(0.0),
    maxErrP = cms.untracked.double(0.1),
    minEta = cms.untracked.double(0.0),
    maxErrQPt = cms.untracked.double(10.0),
    maxErrDz = cms.untracked.double(0.01),
    maxPt = cms.untracked.double(1000.0),
    nTrks = cms.untracked.uint32(50),
    nHits = cms.untracked.uint32(70),
    maxErrPhi = cms.untracked.double(0.003),
    nBinP = cms.untracked.uint32(50),
    maxErrDxy = cms.untracked.double(0.01),
    maxErrEta = cms.untracked.double(0.01),
    minErrQPt = cms.untracked.double(-10.0),
    nBinPhi = cms.untracked.uint32(50),
    maxErrPt = cms.untracked.double(0.1)
)

