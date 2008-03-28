import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    sourceSeed = cms.untracked.uint32(98765)
)

process.source = cms.Source("FlatRandomPtGunSource",
    maxEvents = cms.untracked.int32(2000),
    PGunParameters = cms.untracked.PSet(
        MaxPt = cms.untracked.double(60.0),
        MinPt = cms.untracked.double(60.0),
        # you can request more than 1 particle
        PartID = cms.untracked.vint32(211),
        MaxEta = cms.untracked.double(3.0),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-3.0),
        #
        # phi must be given in radians
        #
        MinPhi = cms.untracked.double(-3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('scan with pion of pt=60GeV'),
    firstRun = cms.untracked.uint32(1)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('Pion_Pt60GeV_all.root')
)

process.p = cms.EndPath(process.GEN)

