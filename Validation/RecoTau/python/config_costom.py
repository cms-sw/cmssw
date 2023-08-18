import os
import FWCore.ParameterSet.Config as cms


process = cms.Process('TauDQMOffline')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000


# import of standard configurations
from Configuration.StandardSequences.GeometryRecoDB_cff import *
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')


process.GlobalTag.globaltag = '94X_dataRun2_ReReco_EOY17_v6'

#process.load("DQMServices.Components.DQMStoreStats_cfi")
#process.load('DQMOffline.Configuration.DQMOffline_cff')


readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

readFiles.extend( [
	'/store/data/Run2017D/Tau/MINIAOD/31Mar2018-v1/00000/02FE19AF-3837-E811-B3FF-44A842B4520B.root'
       ] );

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( -1 )
)

process.load('Validation.RecoTau.RecoTauValidation_cff')
#process.load('Validation.RecoTau.DQMSequences_cfi')
#process.load('Validation.RecoTau.RecoTauValidationMiniAOD_cfi')

process.dump = cms.EDAnalyzer("EventContentAnalyzer")


process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('RECO_RAW2DIGI_L1Reco_RECO_EI_PAT_DQM_inDQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)


#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(8)
process.options.numberOfStreams=cms.untracked.uint32(0)

#Have logErrorHarvester wait for the same EDProducers to finish as those providing data for the OutputModule
from FWCore.Modules.logErrorHarvester_cff import customiseLogErrorHarvesterUsingOutputCommands
process = customiseLogErrorHarvesterUsingOutputCommands(process)

