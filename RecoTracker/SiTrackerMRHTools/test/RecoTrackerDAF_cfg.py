import FWCore.ParameterSet.Config as cms

process = cms.Process("tracking")
process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger = cms.Service("MessageLogger",
#    debugmessages_ttbar_group = cms.untracked.PSet(
#        INFO = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#        ),
#        GroupedDAFHitCollector = cms.untracked.PSet(
#            limit = cms.untracked.int32(10000000)
#        ),
##        SiTrackerMultiRecHitUpdator = cms.untracked.PSet(
##            limit = cms.untracked.int32(10000000)
##        ),
##        DEBUG = cms.untracked.PSet(
##            limit = cms.untracked.int32(0)
##        ),
##        DAFTrackProducerAlgorithm = cms.untracked.PSet(
##            limit = cms.untracked.int32(10000000)
##        ),
##        TrackFitters = cms.untracked.PSet(
##            limit = cms.untracked.int32(10000000)
##        ),
##        threshold = cms.untracked.string('DEBUG')
##    ),
##    debugModules = cms.untracked.vstring('*'),
##    categories = cms.untracked.vstring('SiTrackerMultiRecHitUpdator',
##'DAFTrackProducerAlgorithm'),
###        'SimpleMTFHitCollector'),
##    destinations = cms.untracked.vstring('debugmessages_ttbar_group',
##        'log_ttbar_group')
##)
process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

import RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi
process.DAFTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
process.DAFTrackCandidateMaker = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
process.load("RecoTracker.TrackProducer.CTFFinalFitWithMaterialDAF_cff")

process.load("Validation.RecoTrack.MultiTrackValidator_cff")

process.load("Validation.RecoTrack.cuts_cff")

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")

process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.Timing = cms.Service("Timing")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_2_1_10/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/78E29B63-4699-DD11-AB65-000423D98750.root',
'/store/relval/CMSSW_2_1_10/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/821B138A-4799-DD11-BEE3-000423D987E0.root',
'/store/relval/CMSSW_2_1_10/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/8A258342-FD99-DD11-8633-000423D991D4.root',
'/store/relval/CMSSW_2_1_10/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/D26C6B7C-4799-DD11-95AC-000423D98EA8.root'
))

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    ignoreTotal = cms.untracked.int32(1)
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('tracks.root')
)

process.validation = cms.Sequence(process.cutsTPEffic*process.cutsTPFake*process.multiTrackValidator)
process.p = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits*process.newSeedFromPairs*process.newSeedFromTriplets*process.newCombinedSeeds*process.DAFTrackCandidateMaker*process.ctfWithMaterialTracksDAF*process.validation)
process.DAFTrajectoryBuilder.ComponentName = 'DAFTrajectoryBuilder'
process.DAFTrajectoryBuilder.trajectoryFilterName = 'newTrajectoryFilter'
process.DAFTrackCandidateMaker.SeedProducer = 'newCombinedSeeds'
process.DAFTrackCandidateMaker.TrajectoryBuilder = 'DAFTrajectoryBuilder'
process.DAFTrackCandidateMaker.useHitsSplitting = False
process.DAFTrackCandidateMaker.doSeedingRegionRebuilding = True
process.MRHFittingSmoother.EstimateCut = -1
process.MRHFittingSmoother.MinNumberOfHits = 3
process.multiTrackValidator.outputFile = 'validationPlots_DAF_singlemu_new.root'
process.multiTrackValidator.label = ['ctfWithMaterialTracksDAF']
process.multiTrackValidator.UseAssociators = True
process.TrackAssociatorByHits.UseSplitting = False


