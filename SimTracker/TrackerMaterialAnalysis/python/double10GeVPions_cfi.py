import FWCore.ParameterSet.Config as cms

# generate pi0/pi0* events
source = cms.Source("EmptySource",
    firstLuminosityBlock = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(10)
)

generator = cms.EDProducer('FlatRandomPtGunProducer',
    PGunParameters = cms.PSet(
        # you can request more than 1 particle
        PartID = cms.vint32(211),
        MinEta = cms.double(-4.0),
        MaxEta = cms.double( 4.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double( 3.14159265359),
        MinPt  = cms.double( 9.99),
        MaxPt  = cms.double(10.01)
    ),
    AddAntiParticle = cms.bool(True),
    Verbosity       = cms.untracked.int32(0),
    psethack        = cms.string('pi0/pi0* Pt=10 -4<Eta<4'),
)
