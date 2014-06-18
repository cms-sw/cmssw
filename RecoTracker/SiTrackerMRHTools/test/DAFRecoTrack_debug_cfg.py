import FWCore.ParameterSet.Config as cms

# inspired from RecoTracker/TrackProducer/test/TrackRefit.py
 
process = cms.Process("Refitting")

### standard MessageLoggerConfiguration
process.load("FWCore.MessageService.MessageLogger_cfi")

### Standard Configurations
process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration/StandardSequences/Geometry_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load('Configuration/StandardSequences/MagneticField_cff')

### Conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')
process.GlobalTag.globaltag = 'START71_V1::All'#POSTLS171_V1::All'

### Track Refitter
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.load("RecoTracker.TrackProducer.CTFFinalFitWithMaterialDAF_cff")
process.ctfWithMaterialTracksDAF.TrajectoryInEvent = True
process.ctfWithMaterialTracksDAF.src = 'TrackRefitter'
process.ctfWithMaterialTracksDAF.TrajAnnealingSaving = True
process.MRHFittingSmoother.EstimateCut = -1
process.MRHFittingSmoother.MinNumberOfHits = 3

# debug
process.simpleMultiRecHitCollector.Debug = True
process.siTrackerMultiRecHitUpdator.Debug = True

process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('file:reco_trk_TTbar_13_5evts.root')
    fileNames = cms.untracked.vstring('file:reco_trk_SingleMuPt10_UP15_10evts.root')
) 
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
                               fileName = cms.untracked.string('refit_DAF_SingleMuPt10_UP15_100evts.root')
                               )

process.p = cms.Path(process.MeasurementTrackerEvent*process.TrackRefitter*process.ctfWithMaterialTracksDAF)
process.o = cms.EndPath(process.out)

process.schedule = cms.Schedule(process.p,process.o)

 
