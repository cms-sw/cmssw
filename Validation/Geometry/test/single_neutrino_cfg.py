# cmsRun single_neutrino_cfg.py nEvents=1000

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

process = cms.Process("TestProcess")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.load("IOMC.EventVertexGenerators.VtxSmearedFlat_cfi")
from Configuration.StandardSequences.VtxSmeared import VtxSmeared
#process.load(VtxSmeared['Flat'])
process.load(VtxSmeared['NoSmear'])

process.load("GeneratorInterface.Core.generatorSmeared_cfi")

process.load("Configuration.EventContent.EventContent_cff")

options = VarParsing('analysis')

options.register('nEvents',
                 10000,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.int,
                 "Maximum number of events"
)

options.parseArguments()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.nEvents)
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.VtxSmeared.engineName = cms.untracked.string('HepJamesRandom')
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = cms.untracked.uint32(98765432)

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(14),
        MinEta = cms.double(-5.5),
        MaxEta = cms.double(5.5),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(10.0),
        MaxE   = cms.double(10.0)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('single_neutrino_random.root')
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared)
process.outpath = cms.EndPath(process.o1)

