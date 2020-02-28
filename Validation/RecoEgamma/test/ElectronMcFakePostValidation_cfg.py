
import sys
import os
import FWCore.ParameterSet.Config as cms

from electronValidationCheck_Env import env
cmsEnv = env() # be careful, cmsEnv != cmsenv. cmsEnv is local

cmsEnv.checkSample() # check the sample value
cmsEnv.checkValues()

if cmsEnv.beginTag() == 'Run2_2017':
    from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
    process = cms.Process("electronPostValidation",Run2_2017)
else:
    from Configuration.Eras.Era_Phase2_cff import Phase2
    process = cms.Process('electronPostValidation',Phase2) 

process.DQMStore = cms.Service("DQMStore")
process.load("Validation.RecoEgamma.ElectronMcFakePostValidator_cfi")
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

t1 = os.environ['inputPostFile'].split('.')
localFileInput = os.environ['inputPostFile'].replace(".root", "_a.root") #
# Source
process.source = cms.Source ("PoolSource",fileNames = cms.untracked.vstring("file:" + localFileInput),
secondaryFileNames = cms.untracked.vstring(),)

process.electronMcFakePostValidator.InputFolderName = cms.string("EgammaV/ElectronMcFakeValidator")
process.electronMcFakePostValidator.OutputFolderName = cms.string("EgammaV/ElectronMcFakeValidator")

from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag = os.environ['TEST_GLOBAL_TAG']#+'::All'
process.GlobalTag.globaltag = '93X_upgrade2023_realistic_v2'
#process.GlobalTag.globaltag = '93X_upgrade2023_realistic_v0'
#process.GlobalTag.globaltag = '93X_mc2017_realistic_v1'

process.dqmSaver.workflow = '/electronHistos/' + t1[1] + '/RECO3'
process.dqmsave_step = cms.Path(process.DQMSaver)

process.p = cms.Path(process.EDMtoME * process.electronMcFakePostValidator * process.dqmStoreStats)

# Schedule
process.schedule = cms.Schedule(
                                process.p,
                                process.dqmsave_step,
)
