import FWCore.ParameterSet.Config as cms
import os

process = cms.Process("CLIENT")

process.load("Configuration.StandardSequences.Reconstruction_cff") 
process.load('Configuration.Geometry.GeometryExtended2023D3Reco_cff')
process.load('Configuration.Geometry.GeometryExtended2023D3_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['upgradePLS3']


process.load("Validation.HGCalValidation.HGCalDigiClient_cfi")
process.hgcalDigiClientEE.Verbosity     = 2
process.hgcalDigiClientHEF = process.hgcalDigiClientEE.clone(
    DetectorName = cms.string("HGCalHESiliconSensitive"))
process.hgcalDigiClientHEB = process.hgcalDigiClientEE.clone(
    DetectorName = cms.string("HGCalHEScintillatorSensitive"))


process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

# summary
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) ) ##

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
        'file:./DigiVal_pi.root'
        )
                            )

process.load("Configuration.StandardSequences.EDMtoMEAtRunEnd_cff")
process.dqmSaver.referenceHandling = cms.untracked.string('all')

cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/HGCalValidation/'+'Harvesting/'+str(cmssw_version)
process.dqmSaver.workflow = Workflow

process.load("Validation.HGCalValidation.HGCalDigiClient_cfi")

process.p = cms.Path(process.EDMtoME *
                     process.hgcalDigiClientEE *
                     process.hgcalDigiClientHEF *
                     process.hgcalDigiClientHEB *
                     process.dqmSaver)
