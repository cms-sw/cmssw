import FWCore.ParameterSet.Config as cms

import os

process = cms.Process("CONV")

process.load("Configuration.StandardSequences.Reconstruction_cff")
#process.load("Configuration.StandardSequences.GeometryIdeal_cff")
process.load('Configuration.Geometry.GeometryExtendedReco_cff')
process.load('Configuration.Geometry.GeometryExtended_cff')


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START62_V1::All')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# summary
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:2015_run1.root',
                                                              'file:2015_run2.root'
                                                              )
)

process.load("Configuration.StandardSequences.EDMtoMEAtRunEnd_cff")
process.dqmSaver.referenceHandling = cms.untracked.string('all')

cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/HcalValidation/'+'Harvesting/'+str(cmssw_version)
process.dqmSaver.workflow = Workflow

process.load("Validation.HcalHits.hcalSimHitsClient_cfi")
process.hcalsimhitsClient.outputFile = 'HcalSimHitsHarvestingME_2015.root'

process.p = cms.Path(process.EDMtoME * process.hcalsimhitsClient * process.dqmSaver)


