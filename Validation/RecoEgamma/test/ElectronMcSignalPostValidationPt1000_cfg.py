
import sys
import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("electronPostValidation")

process.DQMStore = cms.Service("DQMStore")
process.load("Validation.RecoEgamma.ElectronMcSignalPostValidatorPt1000_cfi")
process.load("DQMServices.Components.DQMStoreStats_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("Configuration.StandardSequences.EDMtoMEAtJobEnd_cff")
# import DQMStore service
process.load('DQMOffline.Configuration.DQMOffline_cff')

# actually read in the DQM root file
process.load("DQMServices.Components.DQMFileReader_cfi")

from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

t1 = os.environ['TEST_HISTOS_FILE'].split('.')
localFileInput = os.environ['TEST_HISTOS_FILE'].replace(".root", "_a.root") #
# Source
process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring("file:" + localFileInput),
secondaryFileNames = cms.untracked.vstring(),)

process.electronMcSignalPostValidatorPt1000.InputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorPt1000")
process.electronMcSignalPostValidatorPt1000.OutputFolderName = cms.string("EgammaV/ElectronMcSignalValidatorPt1000")

from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = os.environ['TEST_GLOBAL_TAG']#+'::All'

process.dqmSaver.workflow = '/electronHistos/' + t1[1] + '/RECO3'
process.dqmsave_step = cms.Path(process.DQMSaver)

process.p = cms.Path(process.EDMtoME * process.electronMcSignalPostValidatorPt1000 * process.dqmStoreStats)

# Schedule
process.schedule = cms.Schedule(
                                process.p,
                                process.dqmsave_step,
)
