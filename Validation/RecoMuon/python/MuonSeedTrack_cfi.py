import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonUpdatorAtVertex_cff import *
muonSeedTrack = cms.EDFilter("MuonSeedTrack",
    MuonUpdatorAtVertex,
    MuonServiceProxy,
    MuonSeed = cms.InputTag("MuonSeed")
)



