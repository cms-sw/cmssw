import FWCore.ParameterSet.Config as cms

process = cms.Process("tracking")
process.load("FWCore.MessageService.MessageLogger_cfi")
#process.MessageLogger = cms.Service("MessageLogger",
#    debugmessages_ttbar_group = cms.untracked.PSet(
#        INFO = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#        ),
#        SimpleMTFHitCollector = cms.untracked.PSet(
#            limit = cms.untracked.int32(10000000)
#        ),
#        SiTrackerMultiRecHitUpdatorMTF = cms.untracked.PSet(
#            limit = cms.untracked.int32(10000000)
#        ),
#        DEBUG = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#        ),
#        MTFTrackProducerAlgorithm = cms.untracked.PSet(
#            limit = cms.untracked.int32(10000000)
#        ),
#        TrackFitters = cms.untracked.PSet(
#            limit = cms.untracked.int32(10000000)
#        ),
#        threshold = cms.untracked.string('DEBUG')
#    ),
#    debugModules = cms.untracked.vstring('*'),
#    categories = cms.untracked.vstring('SiTrackerMultiRecHitUpdatorMTF',
#'MTFTrackProducerAlgorithm',
#        'SimpleMTFHitCollector'),
#    destinations = cms.untracked.vstring('debugmessages_ttbar_group',
#        'log_ttbar_group')
#)

process.load("Configuration.StandardSequences.Services_cff")

process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'IDEAL_V9::All'

#process.load("Configuration.StandardSequences.Geometry_cff")

#process.load("Configuration.StandardSequences.MagneticField_cff")

#process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

import RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi
process.MTFTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
process.MTFTrackCandidateMaker = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
process.load("RecoTracker.TrackProducer.CTFFinalFitWithMaterialMTF_cff")

process.load("Validation.RecoTrack.MultiTrackValidator_cff")

process.load("Validation.RecoTrack.cuts_cff")

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")

process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")

#process.load("GiulioStuff.MyValidation.DAFValidator_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.Timing = cms.Service("Timing")

process.source = cms.Source("PoolSource",
  fileNames = 
cms.untracked.vstring(
       
'/store/relval/CMSSW_2_1_10/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/78E29B63-4699-DD11-AB65-000423D98750.root',
       
'/store/relval/CMSSW_2_1_10/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/821B138A-4799-DD11-BEE3-000423D987E0.root',
       
'/store/relval/CMSSW_2_1_10/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/8A258342-FD99-DD11-8633-000423D991D4.root',
       
'/store/relval/CMSSW_2_1_10/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/D26C6B7C-4799-DD11-95AC-000423D98EA8.root'
 ))

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    ignoreTotal = cms.untracked.int32(-1)
)

#process.out = cms.OutputModule("PoolOutputModule",
#    outputCommands = cms.untracked.vstring('keep *'),
#    fileName = cms.untracked.string('/tmp/tropiano/tracksMTF_100_ann80.root')
#)
process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
# replace this with whatever track type you want to look at
process.TrackRefitter.TrajectoryInEvent = True

from RecoTracker.FinalTrackSelectors.TracksWithQuality_cff import *
process.withLooseQuality.src = 'ctfWithMaterialTracksMTF'
process.ctfWithMaterialTracksMTF.TrajectoryInEvent = True

process.ctfWithMaterialTracksMTF.src="TrackRefitter"
process.validation = cms.Sequence(process.cutsTPEffic*process.cutsTPFake*process.multiTrackValidator)
#process.p = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits*process.newSeedFromPairs*process.newSeedFromTriplets*process.newCombinedSeeds*process.MTFTrackCandidateMaker*process.ctfWithMaterialTracksMTF*process.tracksWithQuality*process.validation*process.mrhvalidation)
process.p = cms.Path(process.TrackRefitter*process.ctfWithMaterialTracksMTF*process.validation)
#process.outpath = cms.EndPath(process.out)
process.MTFTrajectoryBuilder.ComponentName = 'MTFTrajectoryBuilder'
process.MTFTrajectoryBuilder.trajectoryFilterName = 'newTrajectoryFilter'
process.MTFTrackCandidateMaker.src = 'newCombinedSeeds'
process.MTFTrackCandidateMaker.TrajectoryBuilder = 'MTFTrajectoryBuilder'
process.MTFTrackCandidateMaker.useHitsSplitting = False

process.MRHFittingSmoother.EstimateCut = -1
process.MRHFittingSmoother.MinNumberOfHits = 3
process.multiTrackValidator.outputFile = 'validationPlots_MTF_Singlemupt10_fitter.root'
process.multiTrackValidator.label = ['ctfWithMaterialTracksMTF']
process.multiTrackValidator.UseAssociators = True
#process.mrhvalidation.TrackCollection = 'ctfWithMaterialTracksMTF'
#process.mrhvalidation.output = 'MTF_Val.root'
process.TrackAssociatorByHits.UseSplitting = False
#import copy
#from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
#RelaxedChi2 = copy.deepcopy(Chi2MeasurementEstimator)
#RelaxedChi2.ComponentName = 'RelaxedChi2'
#RelaxedChi2.MaxChi2 = 100.

#process.siTrackerMultiRecHitUpdatorMTF.AnnealingProgram =[1.0]
