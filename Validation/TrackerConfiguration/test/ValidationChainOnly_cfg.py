import FWCore.ParameterSet.Config as cms

##################################################################
# Put here the globaltag the file name and the number of events:

gtag=cms.string('IDEAL_30X::All')

inputfiles=cms.untracked.vstring('/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-RECO/IDEAL_30X_v1/0005/50E9BA78-E9DD-DD11-8AC9-000423D98B08.root')
secinputfiles=cms.untracked.vstring('/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/6EFD547F-E9DD-DD11-B456-000423D99264.root',
'/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/38E34C97-E8DD-DD11-8327-000423D94534.root',
'/store/relval/CMSSW_3_0_0_pre6/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_30X_v1/0005/D6EBDF31-41DE-DD11-91F0-000423D952C0.root')
nevents=cms.untracked.int32(1)
###################################################################

process = cms.Process("TrackerValidationOnly")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.Services_cff")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = gtag

process.load("SimTracker.Configuration.SimTracker_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")

process.load("Validation.TrackerHits.trackerHitsValidation_cff")

process.load("Validation.TrackerDigis.trackerDigisValidation_cff")

process.load("Validation.TrackerRecHits.trackerRecHitsValidation_cff")

process.load("Validation.TrackingMCTruth.trackingTruthValidation_cfi")

process.load("Validation.RecoTrack.TrackValidation_cff")

process.load("Validation.RecoTrack.SiTrackingRecHitsValid_cff")

process.maxEvents = cms.untracked.PSet(
    input = nevents
)
process.source = cms.Source("PoolSource",
    fileNames = inputfiles,
     secondaryFileNames = secinputfiles                          
)

process.Timing = cms.Service("Timing")

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck")

process.PixelTrackingRecHitsValid.src = 'TrackRefitter'
process.StripTrackingRecHitsValid.trajectoryInput = 'TrackRefitter'

process.trackerHitsValid.outputFile='TrackerHitHisto.root'
process.pixelDigisValid.outputFile='pixeldigihisto.root'
process.stripDigisValid.outputFile='stripdigihisto.root'
process.pixRecHitsValid.outputFile='pixelrechitshisto.root'
process.stripRecHitsValid.outputFile='sistriprechitshisto.root'
process.trackingTruthValid.outputFile='trackingtruthhisto.root'
process.multiTrackValidator.outputFile='validationPlots.root'
process.PixelTrackingRecHitsValid.outputFile='pixeltrackingrechitshist.root'
process.StripTrackingRecHitsValid.outputFile='striptrackingrechitshisto.root'

process.simhits = cms.Sequence(process.trackerHitsValidation)
process.digis = cms.Sequence(process.trackerDigisValidation)
process.rechits = cms.Sequence(process.siPixelRecHits*process.siStripMatchedRecHits*process.trackerRecHitsValidation)
process.tracks = cms.Sequence(process.trackingTruthValid*process.tracksValidation)
process.trackinghits = cms.Sequence(process.TrackRefitter*process.trackingRecHitsValid)
process.p1 = cms.Path(process.mix*process.simhits*process.digis*process.rechits*process.tracks*process.trackinghits)
