import FWCore.ParameterSet.Config as cms

# generate single nu_mu events
generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        # you can request more than 1 particle
        PartID = cms.vint32(14),
        MinEta = cms.double(-4.0),
        MaxEta = cms.double( 4.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double( 3.14159265359),
        MinPt  = cms.double( 9.99),
        MaxPt  = cms.double(10.01)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0),
    firstRun        = cms.untracked.uint32(1)
)
