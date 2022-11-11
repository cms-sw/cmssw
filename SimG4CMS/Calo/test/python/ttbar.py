###############################################################################
# Way to use this:
#   cmsRun ttbar.py geometry=2016
#   Options for geometry 2016, 2017, 2018, 2021, 2026, legacy
# 
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "2021",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: 2016, 2017, 2018, 2021, 2026, legacy")

### get and parse the command line arguments
 
options.parseArguments()

print(options)

####################################################################
# Use the options
histFile = "ttbar" + options.geometry + ".root"
outFile = "simevent_ttbar" + options.geometry + ".root"

if (options.geometry == "2016"):
    from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
    process = cms.Process('Sim',Run2_2016)
    geomFile = "Configuration.Geometry.GeometryExtended" + options.geometry + "Reco_cff"
    globalTag = "auto:run2_mc"
elif (options.geometry == "2017"):
    from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
    process = cms.Process('Sim',Run2_2017)
    geomFile = "Configuration.Geometry.GeometryExtended" + options.geometry + "Reco_cff"
    globalTag = "auto:phase1_2017_realistic"
elif (options.geometry == "2018"):
    from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
    process = cms.Process('Sim',Run2_2018)
    geomFile = "Configuration.Geometry.GeometryExtended" + options.geometry + "Reco_cff"
    globalTag = "auto:phase1_2018_realistic"
elif (options.geometry == "2021"):
    from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
    process = cms.Process('Sim',Run3_DDD)
    geomFile = "Configuration.Geometry.GeometryExtended" + options.geometry + "Reco_cff"
    globalTag = "auto:phase1_2022_realistic"
elif (options.geometry == "2026"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('Sim',Phase2C11M9)
    geomFile = "Configuration.Geometry.GeometryExtended" + options.geometry + "D88Reco_cff"
    globalTag = "auto:phase2_realistic"
else:
    process = cms.Process('Sim')
    geomFile = "Configuration.Geometry.GeometryExtendedReco_cff"
    globalTag = "auto:run1_mc"

print("Geometry file: ", geomFile)
print("Hist file:     ", histFile)
print("Output file:   ", outFile)
print("Gobal Tag:     ", globalTag)

process.load("SimG4CMS.Calo.PythiaTT_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")
process.load(geomFile)
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load("SimG4CMS.Calo.CaloSimHitStudy_cfi")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, globalTag, '')

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.HitStudy=dict()
    process.MessageLogger.HcalSim=dict()

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

process.TFileService = cms.Service("TFileService",
    fileName = cms.string(histFile)
)

# Event output
process.output = cms.OutputModule("PoolOutputModule",
    process.FEVTSIMEventContent,
    fileName = cms.untracked.string(outFile)
)

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.analysis_step   = cms.Path(process.CaloSimHitStudy)
process.out_step = cms.EndPath(process.output)

process.generator.pythiaHepMCVerbosity = False
process.generator.pythiaPylistVerbosity = 0
process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT_EMM'
process.g4SimHits.HCalSD.TestNumberingScheme = False
process.CaloSimHitStudy.TestNumbering = False

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
                                process.simulation_step,
                                process.analysis_step,
#                               process.out_step
                                )

# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq
