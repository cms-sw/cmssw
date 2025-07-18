# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: Configuration/GenProduction/python/BPH-RunIIFall18GS-00369-fragment.py --python_filename BPH-RunIIFall18GS-00369_1_cfg.py --eventcontent RAWSIM --datatier GEN-SIM --fileout file:BPH-RunIIFall18GS-00369.root --conditions 102X_upgrade2018_realistic_v11 --beamspot Realistic25ns13TeVEarly2018Collision --step GEN,SIM --geometry DB:Extended --era Run2_2018 --no_exec --mc -n 1000
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from FWCore.ParameterSet.VarParsing import VarParsing

process = cms.Process('SIM',eras.Run3_2023)
options = VarParsing ('analysis')
options.outputFile = 'file:sexaq_sim.root'
#options.inputFiles = 'root://cmsxrootd.hep.wisc.edu//store/user/wvetens/crmc_Sexaq/crmc/Sexaquark_13TeV_trial_4_1p8GeV/0/crmc_Sexaq_1.root'
options.inputFiles = 'file:crmc_Sexaq_1.root'
options.maxEvents= 100
options.parseArguments()

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
#process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
# Vtx Smearing done in hepmc 2 gen step
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic25ns13TeVEarly2018Collision_cfi')
#process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('RecoVertex.BeamSpotProducer.BeamSpot_cfi')

# Lengthy message logs - uncomment to debug
process.MessageLogger = cms.Service("MessageLogger",
  destinations = cms.untracked.vstring('cout'),
  cout = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO')
  )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
    #input = cms.untracked.int32(-1)
)

# Input source

#process.source = cms.Source("EmptySource")
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles),
    skipEvents = cms.untracked.uint32(0),
    duplicateCheckMode = cms.untracked.string ("noDuplicateCheck")
)


process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True)
)

# Output definition

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(1),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(20971520),
    fileName = cms.untracked.string(options.outputFile),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition
process.RAWSIMoutput.outputCommands += ("keep *_genParticlesPlusGEANT_*_*",)

# Other statements
#process.XMLFromDBSource.label = cms.string("Extended")
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, '102X_upgrade2018_realistic_v11', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, '140X_mcRun3_2024_realistic_v14', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '140X_mcRun3_2023_realistic_v3', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, '140X_mcRun3_2022_realistic_v3', '')



process.genParticlesPlusGEANT = cms.EDProducer("GenPlusSimParticleProducer",
  src           = cms.InputTag("g4SimHits"),
  setStatus     = cms.int32(8),                 # set status = 8 for GEANT GPs
  particleTypes = cms.vstring(),
  filter = cms.vstring(),
  genParticles  = cms.InputTag("genParticles") # original genParticle list
)

from SimG4Core.CustomPhysics.CustomPhysics_cfi import customPhysicsSetup
process.g4SimHits.Physics.type = cms.string('SimG4Core/Physics/CustomPhysics')
process.g4SimHits.Physics.RHadronDummyFlip = cms.bool(False)
process.g4SimHits.Physics.Verbosity = 1
process.g4SimHits.Physics = cms.PSet(
  process.g4SimHits.Physics, #keep all default value and add others
  customPhysicsSetup
  )
process.g4SimHits.Physics.particlesDef = cms.FileInPath('SimG4Core/CustomPhysics/data/particles_sexaq_1p8_GeV.txt')


## Vtx Smearing done in hepmc 2 gen step
process.VtxSmeared.src = cms.InputTag("source", "generator")
process.genParticles.src = cms.InputTag("generatorSmeared")
process.g4SimHits.HepMCProductLabel = cms.InputTag("generatorSmeared")
process.g4SimHits.Generator.HepMCProductLabel = cms.InputTag("generatorSmeared")


# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
#moved beamspot and vtx smearing to hepmc2gen step
#process.simulation_step = cms.Path(process.offlineBeamSpot*process.generatorSmeared*process.psim*process.genParticlesPlusGEANT)
process.simulation_step = cms.Path(process.psim*process.genParticlesPlusGEANT)
#process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
#process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.endjob_step,process.RAWSIMoutput_step)
process.schedule = cms.Schedule(process.generation_step,process.simulation_step,process.endjob_step,process.RAWSIMoutput_step)

from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)
## filter all path with the production filter sequence
#for path in process.paths:
#	getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq 

from Configuration.DataProcessing.Utils import addMonitoring
process = addMonitoring(process)

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)

# End adding early deletion
# For debug:
#print process.dumpPython()
