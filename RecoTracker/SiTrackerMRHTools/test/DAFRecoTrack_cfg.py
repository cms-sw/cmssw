import FWCore.ParameterSet.Config as cms

# inspired from RecoTracker/TrackProducer/test/TrackRefit.py
 
process = cms.Process("RefittingDAF")

### standard MessageLoggerConfiguration
process.load("FWCore.MessageService.MessageLogger_cfi")

### Standard Configurations
process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

### Conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')
# 2022
process.GlobalTag.globaltag = 'auto:phase1_2022_realistic'

### Track Refitter
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.load("RecoTracker.TrackProducer.CTFFinalFitWithMaterialDAF_cff")
process.TracksDAF.TrajectoryInEvent = True
process.TracksDAF.src = 'TrackRefitter'
process.TracksDAF.TrajAnnealingSaving = False
process.MRHFittingSmoother.EstimateCut = -1
process.MRHFittingSmoother.MinNumberOfHits = 3

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
) 
process.PoolSource.fileNames = [
  "file:reco_trk.root",
]
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('drop *_*_*_*', 
								      'keep *_siPixelClusters_*_*', 
								      'keep *_siStripClusters_*_*', 
								      'keep *_siPixelDigis_*_*', 
								      'keep *_siStripDigis_*_*', 
								      'keep *_offlineBeamSpot_*_*',
								      'keep *_pixelVertices_*_*',
								      'keep *_siStripMatchedRecHits_*_*', 
								      'keep *_initialStepSeeds_*_*', 
                                                                      'keep recoTracks_*_*_*',
                                                                      'keep recoTrackExtras_*_*_*',
                                                                      'keep TrackingRecHitsOwned_*_*_*'),
                               fileName = cms.untracked.string('refit_DAF.root')
                               )

process.p = cms.Path(process.MeasurementTrackerEvent*process.TrackRefitter*process.TracksDAF)
process.o = cms.EndPath(process.out)

## debug(DAFTrackProducerAlgorithm)
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring("debugTracking_DAF"), 
                                    debugModules = cms.untracked.vstring("*"), 
                                    categories = cms.untracked.vstring("DAFTrackProducerAlgorithm"),
                                    debugTracking_DAF = cms.untracked.PSet(threshold = cms.untracked.string("DEBUG"), 
                                                                      DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)), 
                                                                      default = cms.untracked.PSet(limit = cms.untracked.int32(0)), 
                                                                      DAFTrackProducerAlgorithm = cms.untracked.PSet(limit = cms.untracked.int32(-1)), 
                                                                       )
                                    )

process.schedule = cms.Schedule(process.p,process.o)


 
