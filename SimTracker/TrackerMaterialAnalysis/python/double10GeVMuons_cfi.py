import FWCore.ParameterSet.Config as cms

# generate mu+/mu- events
source = cms.Source("FlatRandomPtGunSource",
    PGunParameters = cms.untracked.PSet(
        # you can request more than 1 particle
        PartID = cms.untracked.vint32(13),
        MinEta = cms.untracked.double(-4.0),
        MaxEta = cms.untracked.double( 4.0),
        MinPhi = cms.untracked.double(-3.14159265359),
        MaxPhi = cms.untracked.double( 3.14159265359),
        MinPt  = cms.untracked.double( 9.99),
        MaxPt  = cms.untracked.double(10.01)
    ),
    AddAntiParticle = cms.untracked.bool(True),
    Verbosity       = cms.untracked.int32(0),
    firstRun        = cms.untracked.uint32(1)
)
