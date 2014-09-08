# The following comments couldn't be translated into the new config version:

#! /bin/env cmsRun

import FWCore.ParameterSet.Config as cms

process = cms.Process("bdebug")

#keep the logging output to a nice level
process.load("FWCore.MessageLogger.MessageLogger_cfi")




# load the full reconstraction configuration, to make sure we're getting all needed dependencies
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

#parallel processing

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
#     skipEvents = cms.untracked.uint32(281),	
    fileNames = cms.untracked.vstring(),
    secondaryFileNames = cms.untracked.vstring()
)

process.GlobalTag.globaltag = 'PRE_STA71_V4::All'


process.load("SimTracker.TrackHistory.TrackClassifier_cff")

process.bTracksProducer = cms.EDProducer("BTracksProducer",
    trackConfig = cms.PSet(process.trackClassifier),
    simG4 = cms.InputTag("g4SimHits"),
    trackingTruth = cms.untracked.InputTag("mix","MergedTrackTruth"),
    trackInputTag = cms.untracked.InputTag("generalTracks"),
    allSim = cms.untracked.bool(True)
)

process.p = cms.Path(process.bTracksProducer)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('bdebug.root'),
)
process.endpath= cms.EndPath(process.out)


process.PoolSource.fileNames = [
"file:trk_00.root",
"file:trk_01.root",
"file:trk_02.root",
"file:trk_03.root",
"file:trk_04.root",
"file:trk_05.root",
"file:trk_06.root",
"file:trk_07.root",
"file:trk_08.root",
"file:trk_09.root",
"file:trk_10.root",
"file:trk_11.root",
"file:trk_12.root",
"file:trk_13.root",
"file:trk_14.root",
]

