###############################################################################
# Way to use this:
#   cmsRun zdcRun3_cfg.py type=Standard
#
#   Options for type Standard, Forward, Legacy
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('type',
                 "Standard",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: Standard, Forward, Legacy")

### get and parse the command line arguments
options.parseArguments()
print(options)

from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD

process = cms.Process("ZDCRun3",Run3_DDD)

if (options.type == "Forward"):
    geomFile = "Geometry.ForwardCommonData.GeometryExtended2021Reco_cff"
else:
    geomFile = "Configuration.Geometry.GeometryExtended2021Reco_cff"
outFile = "zdc" + options.type + ".root"
globalTag = "auto:phase1_2022_realistic"

print("Geometry file: ", geomFile)
print("Global Tag:    ", globalTag)
print("Output file:   ", outFile)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load(geomFile)
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, globalTag, '')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.ForwardSim=dict()
    process.MessageLogger.ZdcSD=dict()

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(2112),
        MinEta = cms.double(8.20),
        MaxEta = cms.double(9.40),
        MinPhi = cms.double(-3.1415926),
        MaxPhi = cms.double(3.1415926),
        MinE   = cms.double(5000.00),
        MaxE   = cms.double(10000.00)
    ),
    Verbosity       = cms.untracked.int32(0),
    AddAntiParticle = cms.bool(True)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    showMallocInfo = cms.untracked.bool(True),
    dump = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789
process.rndmStore = cms.EDProducer("RandomEngineStateProducer")

# Event output
process.output = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string(outFile)
)

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.out_step = cms.EndPath(process.output)

process.g4SimHits.Generator.MinEtaCut = -10.0
process.g4SimHits.Generator.MaxEtaCut = 10.0
process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.LHCTransport = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMM'
process.g4SimHits.EventVerbose = 2
process.g4SimHits.SteppingVerbosity = 2
process.g4SimHits.StepVerboseThreshold= 0.1
process.g4SimHits.VerboseEvents = [1,2,3,4,5]
process.g4SimHits.VertexNumber = []
process.g4SimHits.VerboseTracks =[]


# Schedule definition                                                          
process.schedule = cms.Schedule(process.generation_step,
                                process.simulation_step,
                                process.out_step
                                )

# filter all path with the production filter sequence                          
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq

