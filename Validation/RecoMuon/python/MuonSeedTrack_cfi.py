import FWCore.ParameterSet.Config as cms

from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonUpdatorAtVertex_cff import *
muonSeedTrack = cms.EDAnalyzer("MuonSeedTrack",
    MuonUpdatorAtVertex,
    MuonServiceProxy,
    MuonSeed = cms.InputTag("MuonSeed")
)



# foo bar baz
# aw7elgVm5yBCw
# aDeUBJ7A0Fwd7
