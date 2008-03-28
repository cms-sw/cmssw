import FWCore.ParameterSet.Config as cms

# generate single-particle events
source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        MaxPt = cms.untracked.double(10.01),
        MinPt = cms.untracked.double(9.99),
        # you can request more than 1 particle
        PartID = cms.untracked.vint32(14),
        MaxEta = cms.untracked.double(4.0),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-4.0),
        MinPhi = cms.untracked.double(-3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0),
    AddAntiParticle = cms.untracked.bool(False),
    firstRun = cms.untracked.uint32(1)
)


