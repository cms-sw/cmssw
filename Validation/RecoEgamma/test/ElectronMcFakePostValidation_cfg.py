
import sys
import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("electronPostValidation")

process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)

t1 = os.environ['TEST_HISTOS_FILE'].split('.')
localFileInput = 'DQM_V0001_R000000001__electronHistos__' + t1[1] + '__RECO.root'
localFileInput2 = 'DQM_V0001_R000000001__electronHistos__' + t1[1] + '__RECO2.root' # temp
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring("file:" + localFileInput))
#process.source = cms.Source("DQMRootSource", fileNames = cms.untracked.vstring("file:" + localFileInput))

process.load("Validation.RecoEgamma.ElectronMcFakePostValidator_cfi")
# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("Configuration.StandardSequences.EDMtoMEAtJobEnd_cff")

process.electronMcSignalPostValidator.InputFile = localFileInput
process.electronMcSignalPostValidator.OutputFile = localFileInput2 # temp

#process.electronMcSignalPostValidator.InputFolderName = cms.string("EgammaV/ElectronMcSignalValidator")
#process.electronMcSignalPostValidator.OutputFolderName = cms.string("EgammaV/ElectronMcSignalValidator")
process.electronMcSignalPostValidator.InputFolderName = cms.string("Run 1/EgammaV/Run summary/ElectronMcSignalValidator")
process.electronMcSignalPostValidator.OutputFolderName = cms.string("Run 1/EgammaV/Run summary/ElectronMcSignalValidator")

process.dqmSaver.workflow = '/electronHistos/' + t1[1] + '/RECO3'
#
process.dqmsave_step = cms.Path(process.DQMSaver)
#process.p = cms.Path(process.electronMcSignalPostValidator*process.dqmStoreStats*process.DQMSaver)
#process.p = cms.Path(process.electronMcSignalPostValidator*process.dqmStoreStats)
process.p = cms.Path(process.electronMcSignalPostValidator)

# Schedule
process.schedule = cms.Schedule(process.p,
                                process.dqmsave_step,
)


