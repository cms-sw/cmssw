import FWCore.ParameterSet.Config as cms

source = cms.Source("EmptySource")

generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(11),
        MinEta = cms.double(0.0),
        MaxEta = cms.double(0.0),
        MinPhi = cms.double(1.57079632679),
        MaxPhi = cms.double(1.57079632679),
        MinE   = cms.double(10.0),
        MaxE   = cms.double(10.0)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity = cms.untracked.int32(0)
)

# Don't smear our vertex!
VtxSmeared = cms.EDProducer("GaussEvtVtxGenerator",
    src    = cms.InputTag("generator","unsmeared"),
    MeanX  = cms.double(0.0),
    MeanY  = cms.double(-2.0),
    MeanZ  = cms.double(0.0),
    SigmaX = cms.double(0.0),
    SigmaY = cms.double(0.0),
    SigmaZ = cms.double(0.0),
    TimeOffset = cms.double(0.0)
)


