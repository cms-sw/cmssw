# The following comments couldn't be translated into the new config version:

#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("convert")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("DQMServices.Core.DQM_cfg")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)


process.load("DQMServices.Components.EDMtoMEConverter_cfi")

process.plots = cms.Path(process.EDMtoMEConverter * process.dqmSaver)


process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/BJET'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd =cms.untracked.bool(True) 
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
process.PoolSource.fileNames = [
    'file:MEtoEDMConverter_1.root'
#    'file:MEtoEDMConverter_2.root',
#    'file:MEtoEDMConverter_3.root',
#    'file:MEtoEDMConverter_4.root',
#    'file:MEtoEDMConverter_5.root',
#    'file:MEtoEDMConverter_6.root',
#    'file:MEtoEDMConverter_7.root',
#    'file:MEtoEDMConverter_8.root',
#    'file:MEtoEDMConverter_9.root',
#    'file:MEtoEDMConverter_10.root',
#    'file:MEtoEDMConverter_11.root'
    ]
    

