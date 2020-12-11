# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step2 --filein=file:GluGluTo2Jets_M_300_2000_13TeV_exhume_cff_py_GEN_SIM_HECTOR_CTPPS.root --conditions auto:run2_mc -s DIGI:pdigi_valid,DIGI2RAW --datatier GEN-SIM-DIGI-RAW -n 10 --era Run2_2016 --eventcontent FEVTDEBUG --no_exec
import FWCore.ParameterSet.Config as cms
import random


from Configuration.StandardSequences.Eras import eras
process = cms.Process('DIGI2RAW',eras.Run2_2017)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load("CondCore.CondDB.CondDB_cfi")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Geometry.VeryForwardGeometry.geometryRPFromDB_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.RandomNumberGeneratorService.g4SimHits.initialSeed = cms.untracked.uint32(random.randint(0,900000000))
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = cms.untracked.uint32(random.randint(0,900000000))


# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring('file:GluGlu_GEN_SIM_2017.root'),
    inputCommands = cms.untracked.vstring('keep *'), 
    secondaryFileNames = cms.untracked.vstring()
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('GluGlu_DIGI_DIGI2RAW_2017.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands+['keep *_ctpps*_*_*',"keep *_*RP*_*_*",'keep *_LHCTransport_*_*'],
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition
    
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step)
process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
