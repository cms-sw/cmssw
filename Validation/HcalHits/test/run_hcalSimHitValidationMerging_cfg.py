import os
import FWCore.ParameterSet.Config as cms

process = cms.Process("Merge")

#--- limit event reports number by each 100th event
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = cms.untracked.vstring('file:output_seed1.root',
                                      'file:output_seed2.root'
                                      )
)

process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')

cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/HcalValidation/'+'Harvesting/'+str(cmssw_version)
process.dqmSaver.workflow = Workflow

process.p = cms.Path(process.EDMtoME * process.dqmSaver)
