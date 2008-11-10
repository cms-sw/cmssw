import FWCore.ParameterSet.Config as cms
import SimGeneral.HepPDTESSource.pythiapdt_cfi


process = cms.Process("tracking")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Services_cff")

process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'IDEAL_V5::All'


process.load("Configuration.StandardSequences.Reconstruction_cff")

import RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi

import TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedSeeds_cfi

process.load("RecoTracker.TrackProducer.CTFFinalFitWithMaterialDAF_cff")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
# replace this with whatever track type you want to look at
process.TrackRefitter.TrajectoryInEvent = True
process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi")

process.load("Validation.RecoTrack.MultiTrackValidator_cfi")

process.load("Validation.RecoTrack.cuts_cff")

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")

process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
    )
process.Timing = cms.Service("Timing")

process.printList = cms.EDFilter("ParticleListDrawer",
                                 src = cms.InputTag("genParticles"),
                                 maxEventsToPrint = cms.untracked.int32(1)
                                )

##process.MessageLogger = cms.Service("MessageLogger",
##                                     debugmessages_ttbar_group = cms.untracked.PSet(
##     INFO = cms.untracked.PSet(
##     limit = cms.untracked.int32(1000000)
##     ),
##     GroupedDAFHitCollector = cms.untracked.PSet(
##     limit = cms.untracked.int32(10000000)
##     ),
##     SiTrackerMultiRecHitUpdator = cms.untracked.PSet(
##     limit = cms.untracked.int32(10000000)
##     ),
##     DEBUG = cms.untracked.PSet(
##     limit = cms.untracked.int32(1000000)
##     ),
##     DAFTrackProducerAlgorithm = cms.untracked.PSet(
##     limit = cms.untracked.int32(10000000)
##     ),
##     #TrackFitters = cms.untracked.PSet(
##     #limit = cms.untracked.int32(10000000)
##     #),
##     threshold = cms.untracked.string('DEBUG')
##     ),
##                                     debugModules = cms.untracked.vstring('ctfWithMaterialTracksDAF',
##                                                                          'DAFTrackProducer',
##                                                                          'DAFTrackProducerAlgorithm',
##                                                                          ),
##                                     categories = cms.untracked.vstring('SiTrackerMultiRecHitUpdator',
##                                                                        'DAFTrackProducerAlgorithm'),
                                    
## destinations = cms.untracked.vstring('debugmessages_ttbar_group',
##                                      'log_ttbar_group')
##)

process.source = cms.Source("PoolSource",
                            fileNames = 
                            cms.untracked.vstring('/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/1E04FC31-F99A-DD11-94EE-0018F3D096DE.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/30CF1A2C-F99A-DD11-9E3B-001A928116FC.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/4826BC56-319B-DD11-9280-0017312B5F3F.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/4C720416-F89A-DD11-B745-003048769FDF.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/4E40F71D-009B-DD11-9350-001731AF684D.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/54767785-FA9A-DD11-977E-001A92810AE2.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/569F692D-F99A-DD11-B0B1-0018F3D095FA.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/5CC3697A-FA9A-DD11-8208-003048678BB8.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/5CFBA80F-F89A-DD11-BD3F-001A92810AC0.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/64040110-F89A-DD11-9B87-001A92971AD8.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/70E3D52B-F99A-DD11-BCEA-001A928116EE.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/88B41D32-F99A-DD11-AF65-003048767EDF.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/8C4A5988-FA9A-DD11-99C6-0018F3D096EA.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/8CCFCA08-FC9A-DD11-8FDC-001A92971B82.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/92EB778B-FA9A-DD11-8EDA-001A92811702.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/9A69A008-FC9A-DD11-8F08-001A928116B0.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/9E310315-F89A-DD11-8E53-00304876A06D.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/A8F35E7C-FA9A-DD11-981E-00304867918E.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/B0DB2688-FA9A-DD11-A206-0018F3D09614.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/B2AEA28C-FA9A-DD11-B37D-003048769E6D.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/B4DEB210-F89A-DD11-B849-0018F3D096F0.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/C01AB22C-F99A-DD11-B036-001A92971B8A.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/C23D072B-F99A-DD11-A22A-001A92811734.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/C42B5087-FA9A-DD11-9DAC-001A928116BE.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/CE0F5D16-F89A-DD11-AE71-001731AF66A9.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/D2EE0083-FA9A-DD11-9215-003048678B1C.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/E46A5591-FA9A-DD11-9DA4-0017319EB90D.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/F27E1517-FC9A-DD11-80C5-001A92810AB2.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0002/1852F443-299B-DD11-A4B1-001A92971B82.root',
                                                  '/store/relval/CMSSW_2_1_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0002/26F9F2ED-579B-DD11-82E4-001A92971B8A.root'),
#                            skipEvents = cms.untracked.uint32(867)
                            )

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
                                        ignoreTotal = cms.untracked.int32(-1)
                                        )

#process.out = cms.OutputModule("PoolOutputModule",
#                               outputCommands = cms.untracked.vstring('keep *'),
#                               fileName = cms.untracked.string('/tmp/tropiano/tracks_100_ann80-9-4-1-1-1(prova).root')
#                               )
from RecoTracker.FinalTrackSelectors.TracksWithQuality_cff import *
process.withLooseQuality.src = 'ctfWithMaterialTracksDAF'
process.ctfWithMaterialTracksDAF.TrajectoryInEvent = True


process.validation = cms.Sequence(process.cutsTPEffic*process.cutsTPFake*process.multiTrackValidator)

process.p = cms.Path(process.TrackRefitter*process.ctfWithMaterialTracksDAF*process.validation)
#process.p = cms.Path(process.mix*process.TrackRefitter*process.ctfWithMaterialTracksDAF)

#process.outpath = cms.EndPath(process.out)
process.ctfWithMaterialTracksDAF.src = 'TrackRefitter'


process.MRHFittingSmoother.EstimateCut = -1
process.MRHFittingSmoother.MinNumberOfHits = 3
process.multiTrackValidator.outputFile = 'prova_tsoscombined_2events.root'
process.multiTrackValidator.label = ['ctfWithMaterialTracksDAF']
process.multiTrackValidator.UseAssociators = True

#process.siTrackerMultiRecHitUpdator.AnnealingProgram = cms.vdouble(80.0, 9.0, 4.0, 1.0, 1.0, 1.0)

process.TrackAssociatorByHits.UseSplitting = False
import copy
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
RelaxedChi2 = copy.deepcopy(Chi2MeasurementEstimator)
RelaxedChi2.ComponentName = 'RelaxedChi2'
RelaxedChi2.MaxChi2 = 100.

