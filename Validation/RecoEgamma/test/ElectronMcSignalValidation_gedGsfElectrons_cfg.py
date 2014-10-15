
import sys
import os
import DQMOffline.EGamma.electronDataDiscovery as dd
import FWCore.ParameterSet.Config as cms

process = cms.Process("electronValidation")

process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)

t1 = os.environ['TEST_HISTOS_FILE'].split('.')
localFileInput = 'DQM_V0001_R000000001__electronHistos__' + t1[1] + '__RECO.root'

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
#process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring(),secondaryFileNames = cms.untracked.vstring())
process.source = cms.Source("DQMRootSource",fileNames = cms.untracked.vstring())
process.source.fileNames.extend(dd.search())
process.DQMoutput = cms.OutputModule("DQMRootOutputModule",fileName = cms.untracked.string("OUT_step1.root"))

process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("Configuration.StandardSequences.EDMtoMEAtJobEnd_cff") # new 
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag = autoCond[os.environ['TEST_GLOBAL_AUTOCOND']]
process.GlobalTag.globaltag = os.environ['TEST_GLOBAL_TAG']#+'::All'


# FOR DATA REDONE FROM RAW, ONE MUST HIDE IsoFromDeps
# CONFIGURATION

process.load("Validation.RecoEgamma.electronIsoFromDeps_cff")
process.load("Validation.RecoEgamma.ElectronMcSignalValidator_gedGsfElectrons_cfi")

# DQM
#process.dqmSaver.saveAtJobEnd = True
#process.dqmsave_step = cms.Path(process.DQMSaver)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

#process.dqmSaver.workflow = localFileInput

#process.electronMcSignalValidator.OutputFile = cms.string(os.environ['TEST_HISTOS_FILE']) # ne sert plus
#process.electronMcSignalValidator.OutputFolderName = cms.string("Run 1/EgammaV/Run summary/ElectronMcSignalValidator")
process.electronMcSignalValidator.InputFolderName = cms.string("EgammaV/ElectronMcSignalValidator")
process.electronMcSignalValidator.OutputFolderName = cms.string("EgammaV/ElectronMcSignalValidator")

#process.p = cms.Path(process.electronMcSignalValidator*process.dqmStoreStats)
#process.p = cms.Path(process.electronIsoFromDeps*process.electronMcSignalValidator*process.dqmStoreStats)
process.p = cms.Path(process.electronMcSignalValidator)
# Schedule
process.schedule = cms.Schedule(process.p,
#                                process.dqmsave_step,
                                process.DQMoutput_step,
)                               

