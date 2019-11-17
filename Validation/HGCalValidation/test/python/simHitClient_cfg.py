import FWCore.ParameterSet.Config as cms
import os

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9
process = cms.Process('CLIENT',Phase2C9)

process.load("Configuration.StandardSequences.Reconstruction_cff") 
process.load('Configuration.Geometry.GeometryExtended2026D46Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2026D46_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['upgradePLS3']


process.load("Validation.HGCalValidation.HGCalSimHitsClient_cff")

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

# summary
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) ) ## 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:./output_test.root')
                            )

process.load("Configuration.StandardSequences.EDMtoMEAtRunEnd_cff")
process.dqmSaver.referenceHandling = cms.untracked.string('all')

cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/HGCalValidation/'+'Harvesting/'+str(cmssw_version)
process.dqmSaver.workflow = Workflow

process.p = cms.Path(process.EDMtoME *
                     process.hgcalSimHitClientEE *
                     process.hgcalSimHitClientHEF *
                     process.hgcalSimHitClientHEB *
                     process.dqmSaver)
