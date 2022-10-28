import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.Eras.Era_Run3_cff import Run3

options = VarParsing('analysis')
options.register("doSim", True, VarParsing.multiplicity.singleton, VarParsing.varType.bool)
options.register("cmssw", "CMSSW_X_Y_Z", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("globalTag", "tag", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("dataSetTag", "sample", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.parseArguments()

process = cms.Process('HARVESTING',Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring('file:step3_inDQM.root')
)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

# Path and EndPath definitions
if options.doSim:
    process.harvesting_step = cms.Path(process.cscDigiHarvesting)
process.dqmsave_step = cms.Path(process.DQMSaver)
process.endjob_step = cms.EndPath(process.endOfProcess)

process.dqmSaver.workflow = '/{}/{}/{}'.format(options.dataSetTag,options.globalTag,options.cmssw)

# Schedule definition
process.schedule = cms.Schedule()

if options.doSim:
    process.schedule.extend([process.harvesting_step])

process.schedule.extend([process.endjob_step, process.dqmsave_step])
