
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

process.load("Validation.RecoEgamma.ElectronMcSignalPostValidator_cfi")

process.electronMcSignalPostValidator.InputFile = cms.string(os.environ['TEST_HISTOS_FILE'])
process.electronMcSignalPostValidator.OutputFile = cms.string(os.environ['TEST_HISTOS_FILE'])
#process.electronMcSignalPostValidator.InputFolderName = cms.string("Run 1/EgammaV/Run summary/ElectronMcSignalValidator")
#process.electronMcSignalPostValidator.OutputFolderName = cms.string("Run 1/EgammaV/Run summary/ElectronMcSignalValidator")

#process.load("Configuration.StandardSequences.EDMtoMEAtJobEnd_cff")
#process.dqmSaver.workflow = os.environ['DQM_WORKFLOW']
#process.dqmsave_step = cms.Path(process.DQMSaver)
#
#process.p = cms.Path(process.electronMcSignalPostValidator*process.dqmStoreStats*process.DQMSaver)

process.p = cms.Path(process.electronMcSignalPostValidator*process.dqmStoreStats)


