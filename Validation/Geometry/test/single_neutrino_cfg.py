# cmsRun single_neutrino_cfg.py nEvents=100000 etaMin=-6.0 etaMax=6.0 antiPart=0

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

process = cms.Process("TestProcess")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("GeneratorInterface.Core.generatorSmeared_cfi")
process.load("Configuration.EventContent.EventContent_cff")
from Configuration.StandardSequences.VtxSmeared import VtxSmeared
process.load(VtxSmeared['NoSmear'])

options = VarParsing('analysis')

options.register('nEvents',
                 100000,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.int,
                 "Maximum number of events"
)
options.register('etaMin',
                 -6.0,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.float,
                 "Minimum Eta for the neutrinos"
)
options.register('etaMax',
                 6.0,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.float,
                 "Maximum Eta for the neutrinos"
)
options.register('antiPart',
                 0,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.int,
                 "Choice of AntiParticle"
)

options.parseArguments()
print(options)

if (options.antiPart == 0):
    antiPart = False
else:
    antiPart = True
print("antiPart: ", antiPart)

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
        MinEta = cms.double(options.etaMin),
        MaxEta = cms.double(options.etaMax),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(10.0),
        MaxE   = cms.double(10.0)
    ),
    AddAntiParticle = cms.bool(antiPart),
    Verbosity       = cms.untracked.int32(0)
)

process.o1 = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string('single_neutrino_random.root')
)

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared)
process.outpath = cms.EndPath(process.o1)
