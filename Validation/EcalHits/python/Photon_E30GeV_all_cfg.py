import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.untracked.uint32(54321)
)

process.source = cms.Source("FlatRandomEGunSource",
    maxEvents = cms.untracked.int32(2000),
    PGunParameters = cms.untracked.PSet(
        # you can request more than 1 particle
        PartID = cms.untracked.vint32(22),
        MaxEta = cms.untracked.double(3.0),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-3.0),
        MinE = cms.untracked.double(30.0),
        MinPhi = cms.untracked.double(-3.14159265359), ## it must be in radians

        MaxE = cms.untracked.double(30.0)
    ),
    Verbosity = cms.untracked.int32(0), ## for printouts, set it to 1 (or greater)   

    psethack = cms.string('scan with 30GeV photon'),
    firstRun = cms.untracked.uint32(1)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Photon_E30GeV_all.root')
)

process.p = cms.EndPath(process.GEN)

