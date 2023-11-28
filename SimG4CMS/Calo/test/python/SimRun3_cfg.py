###############################################################################
# Way to use this:
#   cmsRun SimRun3_cfg.py geometry=Default type=DDD data=Muon
#
#   Options for geometry: Default, Other
#               type: DDD, DD4hep
#               data: Muon, MinBias
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re, random
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "Default",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: Default, Other")
options.register('type',
                 "DDD",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: DDD, DD4hep")
options.register('data',
                 "Muon",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "data of operations: Muon, MinBias")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

if (options.type == "DDD"):
    from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
    process = cms.Process('SimRun3',Run3_DDD)
    if (options.geometry == "Default"):
        geomFile = "Configuration.Geometry.GeometryExtended2021Reco_cff"
    else:
        geomFile = "Geometry.HcalCommonData.GeometryExtended2021Reco_cff"
else:
    from Configuration.Eras.Era_Run3_cff import Run3
    process = cms.Process('SimRun3',Run3)
    if (options.geometry == "Default"):
        geomFile = "Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff"
    else:
        geomFile = "Geometry.HcalCommonData.GeometryDD4hepExtended2021Reco_cff"

globalTag = "auto:phase1_2022_realistic"
inFile = "file:step0" + options.data + ".root"
outFile = "file:step1" + "Run3" + options.geometry + options.type + options.data + ".root"
tFile = "file:" + "Run3" + options.geometry + options.type + options.data + ".root"

print("Geometry file: ", geomFile)
print("Global Tag:    ", globalTag)
print("Input file:    ", inFile)
print("Output file:   ", outFile)
print("Histogram file:", tFile)

process.load(geomFile)
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.SimIdeal_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, globalTag, '')

process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring(inFile),
    secondaryFileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.HitStudy=dict()
#   process.MessageLogger.SensitiveDetector=dict()

process.Timing = cms.Service("Timing")

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

process.load("SimG4CMS.Calo.CaloSimHitStudy_cfi")
process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(tFile)
)

process.simulation_step = cms.Path(process.psim)
process.out_step = cms.EndPath(process.output)
process.analysis_step = cms.EndPath(process.CaloSimHitStudy)

process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMM'
process.g4SimHits.LHCTransport = False

# Schedule definition
process.schedule = cms.Schedule(process.simulation_step,
                                process.out_step,
                                process.analysis_step,
                                )
