import FWCore.ParameterSet.Config as cms

process = cms.Process("GEN")
# this will run plig-in energy-flat random particle gun
# and puts particles (HepMCPRoduct) into edm::Event
process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        generator = cms.untracked.uint32(456789)
    ),
    sourceSeed = cms.untracked.uint32(54321)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(211),
        MinEta = cms.double(0.3045),
        MaxEta = cms.double(0.3045),
        MinPhi = cms.double(0.3054),
        MaxPhi = cms.double(0.3054),
        MinE   = cms.double(100.0),
        MaxE   = cms.double(100.0)
    ),
    AddAntiParticle = cms.bool(False),
    psethack        = cms.string('single pion 100GeV on barrel'),
    Verbosity       = cms.untracked.int32(0),
    firstRun        = cms.untracked.uint32(1)
)

process.GEN = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('mc_pi+100_etaphi44.root')
)

process.p1 = cms.Path(process.generator)
process.p2 = cms.EndPath(process.GEN)

