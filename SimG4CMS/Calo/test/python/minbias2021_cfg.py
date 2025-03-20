###############################################################################
# Way to use this:
#   cmsRun minbias2021_cfg.py type=DDD var=ZeroMaterial
#
#   Options for type: DDD, DD4hep
#           for var: ZeroMaterial, FlatMinus05Percent, FlatMinus10Percent,
#                    FlatPlus05Percent, FlatPlus10Percent
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re, random
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('type',
                 "DD4hep",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: DDD, DD4hep")
options.register('var',
                 "",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "var of operations: ZeroMaterial, FlatMinus05Percent, FlatMinus10Percent, FlatPlus05Percent, FlatPlus10Percent")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options
if (options.type == "DD4hep"):
    from Configuration.Eras.Era_Run3_cff import Run3
    process = cms.Process("Sim",Run3)
    geomfile = "Configuration.Geometry.GeometryDD4hepExtended2021" + options.var + "Reco_cff"
    outfile = "minbias_FTFP_BERT_EMM_" + options.var + "_DD4hep.root"
else:
    from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
    process = cms.Process("Sim",Run3_DDD)
    geomfile = "Configuration.Geometry.GeometryExtended2021" + options.var + "Reco_cff"
    outfile = "minbias_FTFP_BERT_EMM_" + options.var + "_DDD.root"

print("Geometry file: ", geomfile)
print("Output file:   ", outfile)

process.load("SimG4CMS.Calo.PythiaMinBias_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load(geomfile)
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.SimG4CoreApplication=dict()

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2)
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(True),
    showMallocInfo = cms.untracked.bool(True),
    ignoreTotal = cms.untracked.int32(1)
)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789
process.rndmStore = cms.EDProducer("RandomEngineStateProducer")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('minbias_FTFP_BERT_EMM.root')
)

# Event output
process.output = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string(outfile)
)

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.out_step = cms.EndPath(process.output)

process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMM'

# Schedule definition                                                          
process.schedule = cms.Schedule(process.generation_step,
                                process.simulation_step,
                                process.out_step
                                )

# filter all path with the production filter sequence                          
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq

