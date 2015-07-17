import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("FlatRandomPtGunProducer",
  PGunParameters = cms.PSet(
    MaxPt = cms.double(50.0),
    MinPt = cms.double(2.0),
    PartID = cms.vint32(-13),
    MinPhi = cms.double(-3.14159265359),
    MaxPhi = cms.double(3.14159265359),
    MinEta = cms.double(-4.5),
    MaxEta = cms.double(4.5)
  ),
  Verbosity = cms.untracked.int32(0),
  psethack = cms.string('single mu pt 40 forward'),
  AddAntiParticle = cms.bool(True),
  firstRun = cms.untracked.uint32(1)
)

