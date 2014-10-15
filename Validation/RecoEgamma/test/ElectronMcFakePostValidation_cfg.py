
import sys
import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("electronPostValidation")

process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))
process.source = cms.Source("EmptySource")

process.load("Validation.RecoEgamma.ElectronMcFakePostValidator_cfi")

t1 = os.environ['TEST_HISTOS_FILE'].split('.')
process.electronMcFakePostValidator.InputFile = 'DQM_V0001_R000000001__electronHistos__' + t1[1] + '__RECO.root'
#process.electronMcFakePostValidator.InputFile = cms.string(os.environ['TEST_HISTOS_FILE'])
process.electronMcFakePostValidator.OutputFile = cms.string(os.environ['TEST_HISTOS_FILE'])
process.electronMcFakePostValidator.InputFolderName = cms.string("Run 1/EgammaV/Run summary/ElectronMcSignalValidator")
process.electronMcFakePostValidator.OutputFolderName = cms.string("Run 1/EgammaV/Run summary/ElectronMcSignalValidator")

process.load("Configuration.StandardSequences.EDMtoMEAtJobEnd_cff")
process.dqmSaver.workflow = '/electronHistos/' + t1[1] + '/RECO'
process.dqmsave_step = cms.Path(process.DQMSaver)
#
#process.p = cms.Path(process.electronMcFakePostValidator*process.dqmStoreStats*process.DQMSaver)
process.p = cms.Path(process.electronMcFakePostValidator*process.dqmStoreStats)

# Schedule
process.schedule = cms.Schedule(process.p,
                                process.dqmsave_step,
)


