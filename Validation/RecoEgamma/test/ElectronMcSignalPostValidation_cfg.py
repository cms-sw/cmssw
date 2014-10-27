
import sys
import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("electronPostValidation")

process.DQMStore = cms.Service("DQMStore")
process.load("Validation.RecoEgamma.ElectronMcSignalPostValidator_cfi")
process.load("DQMServices.Components.DQMStoreStats_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
#process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff') # NEW
process.load("Configuration.StandardSequences.EDMtoMEAtJobEnd_cff")


from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)

t1 = os.environ['TEST_HISTOS_FILE'].split('.')
localFileInput = 'DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root' # 'DQM_V0001_R000000001__electronHistos__' + t1[1] + '__RECO.root'
localFileInput2 = 'DQM_V0001_R000000001__electronHistos__' + t1[1] + '__RECO2.root' # temp
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

#process.electronMcSignalPostValidator.InputFolderName = cms.string("EgammaV/ElectronMcSignalValidator")
#process.electronMcSignalPostValidator.OutputFolderName = cms.string("EgammaV/ElectronMcSignalValidator")
process.electronMcSignalPostValidator.InputFolderName = cms.string("Run 1/EgammaV/Run summary/ElectronMcSignalValidator")
process.electronMcSignalPostValidator.OutputFolderName = cms.string("Run 1/EgammaV/Run summary/ElectronMcSignalValidator")

# Source
process.source = cms.Source("EmptySource")
#process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring("file:" + localFileInput))
#process.source = cms.Source("DQMRootSource", fileNames = cms.untracked.vstring(localFileInput))

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = os.environ['TEST_GLOBAL_TAG']#+'::All'

print "localFileInput : input :", localFileInput
print "localFileInput2 : ouput : ", localFileInput2
process.electronMcSignalPostValidator.InputFile = localFileInput
process.electronMcSignalPostValidator.OutputFile = localFileInput2 # temp

process.dqmSaver.workflow = '/electronHistos/' + t1[1] + '/RECO3'
#
process.dqmsave_step = cms.Path(process.DQMSaver)

#process.p = cms.Path(process.electronMcSignalPostValidator*process.dqmStoreStats)
process.p = cms.Path(process.electronMcSignalPostValidator)

# Schedule
process.schedule = cms.Schedule(process.p,
                                process.dqmsave_step,
)

