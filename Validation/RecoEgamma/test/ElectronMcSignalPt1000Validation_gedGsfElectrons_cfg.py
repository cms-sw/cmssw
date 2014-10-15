
import sys
import os
import DQMOffline.EGamma.electronDataDiscovery as dd
import FWCore.ParameterSet.Config as cms

process = cms.Process("electronValidation")

process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring(),secondaryFileNames = cms.untracked.vstring())
process.source.fileNames.extend(dd.search())

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
process.load("Configuration.StandardSequences.EDMtoMEAtJobEnd_cff") # new voir si on garde
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# load DQM
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag = autoCond[os.environ['TEST_GLOBAL_AUTOCOND']]
# next line is for old releases, prior to 7XX
process.GlobalTag.globaltag = os.environ['TEST_GLOBAL_TAG']#+'::All'
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')  #for MC
#print "globaltag : ", os.environ['TEST_GLOBAL_TAG'] # sort POSTLS172_V3


# FOR DATA REDONE FROM RAW, ONE MUST HIDE IsoFromDeps
# CONFIGURATION

process.load("Validation.RecoEgamma.electronIsoFromDeps_cff")
process.load("Validation.RecoEgamma.ElectronMcSignalPt1000Validator_gedGsfElectrons_cfi")

# DQM
process.dqmSaver.saveAtJobEnd = True
#process.dqmSaver.convention = 'Offline'
#process.dqmSaver.referenceHandling = cms.untracked.string('all')
process.dqmsave_step = cms.Path(process.DQMSaver)
#process.edmtome_step = cms.Path(process.EDMtoME) # new voir si on garde
#process.DQMoutput_step = cms.EndPath( process.DQMoutput )

process.electronMcSignalValidator.OutputFile = cms.string(os.environ['TEST_HISTOS_FILE'])
#process.electronMcSignalValidator.OutputFolderName = cms.string("Run 1/EgammaV/Run summary/ElectronMcSignalValidator")

t1 = os.environ['TEST_HISTOS_FILE'].split('.')
process.dqmSaver.workflow = '/electronHistos/' + t1[1] + '/RECO'

process.p = cms.Path(process.electronMcSignalValidator*process.dqmStoreStats)
#process.p = cms.Path(process.electronIsoFromDeps*process.electronMcSignalValidator*process.dqmStoreStats)

# Schedule
#process.schedule = cms.Schedule(process.edmtome_step, process.p,
process.schedule = cms.Schedule(process.p,
                                process.dqmsave_step,
#                                process.DQMoutput_step
)



