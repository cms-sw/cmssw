import FWCore.ParameterSet.Config as cms

# Modified from Configuration/Generator/python/SingleMuPt10_cfi.py
generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(50.0),
        MinPt = cms.double(0.9),
        PartID = cms.vint32(111,111),
        MaxEta = cms.double(2.5),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-2.5),
        MinPhi = cms.double(-3.14159265359) ## in radians

    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('Four Pi0 pt 1 to 50'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)
