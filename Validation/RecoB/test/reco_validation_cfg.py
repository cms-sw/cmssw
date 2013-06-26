# The following comments couldn't be translated into the new config version:

#! /bin/env cmsRun

#file used in the past to run on RECO files, now used for harvesting only on DQM files
import FWCore.ParameterSet.Config as cms

runOnMC = True

process = cms.Process("harvest")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.load("DQMServices.Core.DQM_cfg")

process.load("RecoBTag.Configuration.RecoBTag_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.load("DQMOffline.RecoB.dqmCollector_cff")
if runOnMC:
    process.dqmSeq = cms.Sequence(process.bTagCollectorSequenceMC * process.dqmSaver)
else:
    process.dqmSeq = cms.Sequence(process.bTagCollectorSequenceDATA * process.dqmSaver)

process.load("DQMServices.Components.EDMtoMEConverter_cfi")
process.plots = cms.Path(process.EDMtoMEConverter * process.dqmSeq)

process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/BJET'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd =cms.untracked.bool(True) 
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)

process.PoolSource.fileNames = [

]

