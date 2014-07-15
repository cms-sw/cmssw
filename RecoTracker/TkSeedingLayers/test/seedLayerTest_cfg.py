import FWCore.ParameterSet.Config as cms

process = cms.Process("TKSEEDING")

# message logger
#process.MessageLogger = cms.Service("MessageLogger",
#     default = cms.untracked.PSet( limit = cms.untracked.int32(10) )
#)

#Adding SimpleMemoryCheck service:
process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                   ignoreTotal=cms.untracked.int32(1),
                                   oncePerEventMode=cms.untracked.bool(False)
)


# source
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( ['file:step3.root' ])
secFiles.extend( ['file:step2.root'] )

process.source = source
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'STARTUP3X_V14::All'

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')

### standard includes
process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration.StandardSequences.Geometry_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.testSeedingLayers = cms.EDAnalyzer("TestSeedingLayers",
)


process.clustToHits = cms.Sequence(
    process.siPixelRecHits*process.siStripMatchedRecHits
)

process.tracking = cms.Sequence(
     process.MixedLayerTriplets*process.MixedLayerPairs
#    process.MeasurementTrackerEvent* # unclear where to put this
#    process.trackingGlobalReco
)


# paths
process.trk = cms.Path(
      process.clustToHits *
      process.tracking *
      process.testSeedingLayers
)


process.schedule = cms.Schedule(
      process.trk
)

process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))
