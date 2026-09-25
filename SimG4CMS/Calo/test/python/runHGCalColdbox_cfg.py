import FWCore.ParameterSet.Config as cms
import os, sys, re, random
import FWCore.ParameterSet.VarParsing as VarParsing


from Configuration.Eras.Era_Phase2C26I13M9_cff import Phase2C26I13M9
from Configuration.Eras.Modifier_phase2_hgcalOnly_cff import phase2_hgcalOnly
from Configuration.Eras.Modifier_phase2_hgcalV19_cff import phase2_hgcalV19
from Configuration.ProcessModifiers.hgcalColdBox_cff import hgcalColdBox

#process = cms.Process("PROD",Phase2C17I13M9,phase2_hgcalOnly,phase2_hgcalV18)

process = cms.Process('SingleMuonSim',Phase2C26I13M9,phase2_hgcalOnly,phase2_hgcalV19,hgcalColdBox)

geomFile = "Geometry.HGCalCommonData.testHGCalV19nO_zmReco_cff"
globalTag = "auto:phase2_realistic_T35_13TeV"
outFile = "file:step1V19nO_zmmu.root"

print("Geometry file: ", geomFile)
print("Global Tag:    ", globalTag)
print("Output file:   ", outFile)

# import of standard configurations
process.load(geomFile)
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.MagneticField_0T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.VtxSmearedNoSmear_cff')
process.load('Configuration.StandardSequences.SimNOBEAM_cff')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('SimG4CMS.Calo.hgcalHitPartial_cff')
process.load('SimG4CMS.Calo.hgcalHitCheck_cff')
process.load("IOMC.RandomEngine.IOMC_cff")

rndm = random.randint(0,200000)
process.RandomNumberGeneratorService.generator.initialSeed = 1234
print("Processing with random number seed: ", 1234)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

process.MessageLogger.cerr.FwkReport.reportEvery = 1

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalError=dict()
    process.MessageLogger.HGCSim=dict()
    process.MessageLogger.HGCalSim=dict()
    process.MessageLogger.CaloSim=dict()
    process.MessageLogger.HGCalGeom=dict()
    
# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfStreams = cms.untracked.uint32(0),
    numberOfThreads = cms.untracked.uint32(1),
    printDependencies = cms.untracked.bool(False),
    sizeOfStackForThreadsInKB = cms.optional.untracked.uint32,
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string(''),
    annotation = cms.untracked.string(''),
    name = cms.untracked.string('Applications')
)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW-RECO')
    ),
    fileName = cms.untracked.string(outFile),
    outputCommands = process.FEVTSIMEventContent.outputCommands+['drop *_*_HGCDigisHEback_*','drop *_*_HGCDigisHEfront_*'],
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, globalTag, '')

process.generator = cms.EDFilter("Pythia8PtGun",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(100.01),
        MinPt = cms.double(99.99),
        ParticleID = cms.vint32(-13),
        AddAntiParticle = cms.bool(True),
        MaxEta = cms.double(2.5),
        MaxPhi = cms.double(3.14159265359*5./6.),
        MinEta = cms.double(1.),
        MinPhi = cms.double(3.14159265359/6.) ## in radians
        ),
        Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts
        psethack = cms.string('single mu pt 100'),
        firstRun = cms.untracked.uint32(1),
        PythiaParameters = cms.PSet(parameterSets = cms.vstring())
)


#Modified to produce hgceedigis
process.ProductionFilterSequence = cms.Sequence(process.generator)

process.g4SimHits.HGCSD.CheckID = False
process.g4SimHits.HGCScintSD.CheckID = False

#process.g4SimHits.HGCSD.HitCollection = 1

#process.hgcalHitCheckEE.verbosity = 2

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.analysis_step = cms.Path(process.hgcalHitCheckEE)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,
                                process.genfiltersummary_step,
				process.simulation_step,
                                process.endjob_step,
                                process.analysis_step,
				process.out_step
				)

#from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
#associatePatAlgosToolsTask(process)
# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path).insert(0, process.ProductionFilterSequence)
