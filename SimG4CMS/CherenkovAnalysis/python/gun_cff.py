import FWCore.ParameterSet.Config as cms

source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        PartID = cms.untracked.vint32(11),
        MaxEta = cms.untracked.double(0.0),
        MaxPhi = cms.untracked.double(1.57079632679),
        MinEta = cms.untracked.double(0.0),
        MinE = cms.untracked.double(10.0),
        MinPhi = cms.untracked.double(1.57079632679),
        MaxE = cms.untracked.double(10.0)
    ),
    Verbosity = cms.untracked.int32(0)
)

# Don't smear our vertex!
VtxSmeared = cms.EDFilter("GaussEvtVtxGenerator",
    MeanX = cms.double(0.0),
    MeanY = cms.double(-2.0),
    MeanZ = cms.double(0.0),
    SigmaY = cms.double(0.0),
    SigmaX = cms.double(0.0),
    SigmaZ = cms.double(0.0),
    TimeOffset = cms.double(0.0)
)


