import FWCore.ParameterSet.Config as cms
import SimGeneral.HepPDTESSource.pythiapdt_cfi


process = cms.Process("tracking")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.Services_cff")

process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'IDEAL_31X::All'


process.load("Configuration.StandardSequences.Reconstruction_cff")

import RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi

import TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedSeeds_cfi

process.load("RecoTracker.TrackProducer.CTFFinalFitWithMaterialDAF_cff")

process.load("SimGeneral.MixingModule.mixNoPU_cfi")

process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
# replace this with whatever track type you want to look at
process.TrackRefitter.TrajectoryInEvent = True
process.RKFittingSmoother.MinNumberOfHits=3
process.TrackRefitter.Fitter = "RKFittingSmoother"

process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi")

process.load("Validation.RecoTrack.MultiTrackValidator_cfi")

process.load("Validation.RecoTrack.cuts_cff")

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")

process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5)
    )
process.Timing = cms.Service("Timing")

process.printList = cms.EDFilter("ParticleListDrawer",
                                 src = cms.InputTag("genParticles"),
                                 maxEventsToPrint = cms.untracked.int32(1)
                                )

#process.MessageLogger = cms.Service("MessageLogger",
#                                     debugmessages_ttbar_group = cms.untracked.PSet(
#     INFO = cms.untracked.PSet(
#     limit = cms.untracked.int32(1000000)
#     ),
#     GroupedDAFHitCollector = cms.untracked.PSet(
#     limit = cms.untracked.int32(10000000)
#     ),
#     SiTrackerMultiRecHitUpdator = cms.untracked.PSet(
#     limit = cms.untracked.int32(10000000)
#     ),
#     DEBUG = cms.untracked.PSet(
#     limit = cms.untracked.int32(1000000)
#     ),
#     DAFTrackProducerAlgorithm = cms.untracked.PSet(
#     limit = cms.untracked.int32(10000000)
#     ),
#     #TrackFitters = cms.untracked.PSet(
#     #limit = cms.untracked.int32(10000000)
#     #),
#     threshold = cms.untracked.string('DEBUG')
#     ),
#                                     debugModules = cms.untracked.vstring('ctfWithMaterialTracksDAF',
#                                                                          'DAFTrackProducer',
#                                                                          'DAFTrackProducerAlgorithm',
#                                                                          ),
#                                     categories = cms.untracked.vstring('SiTrackerMultiRecHitUpdator',
#                                                                        'DAFTrackProducerAlgorithm'),
#                                    
# destinations = cms.untracked.vstring('debugmessages_ttbar_group',
#                                      'log_ttbar_group')
#)

process.source = cms.Source("PoolSource",
                            fileNames = 
cms.untracked.vstring([
        '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_31X_v1/0000/0A292B4F-412C-DE11-8D2C-000423D992DC.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_31X_v1/0000/30E54A06-D42B-DE11-BC53-001617E30CC8.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_31X_v1/0000/5692DEC2-972B-DE11-8D46-000423D98844.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_31X_v1/0000/9233C112-9C2B-DE11-B020-001617E30D0A.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_31X_v1/0000/F85031B4-D92B-DE11-91CF-0019DB29C5FC.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-RECO/IDEAL_31X_v1/0000/FEF32BB5-9E2B-DE11-9697-0019B9F6C674.root'   

]),
secondaryFileNames =cms.untracked.vstring( [
           '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/18A5F0EF-CE2B-DE11-A86B-000423D998BA.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/1EB8042C-C52B-DE11-BD17-000423D98804.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/20A47495-992B-DE11-97D1-000423D9880C.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/221AB904-C82B-DE11-A56A-000423D99896.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/24F72B84-D32B-DE11-95D5-000423D98844.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/40009F9C-EB2B-DE11-8585-001617E30F48.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/4C3E20F7-0B2C-DE11-B546-001617E30CD4.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/4C993B39-9A2B-DE11-821A-0019DB29C620.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/52F078F8-9A2B-DE11-8C45-001617C3B6FE.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/56F55BFC-D72B-DE11-9E86-000423D991F0.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/62CF9580-CA2B-DE11-8F57-001617E30CD4.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/64DE49F9-D22B-DE11-A02D-000423D99CEE.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/66B68364-CC2B-DE11-939E-001617DBD5B2.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/70FF520F-D22B-DE11-BD57-000423D98EC8.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/72C04553-972B-DE11-91D9-000423D98B5C.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/84BAEE29-982B-DE11-8E6C-000423D98A44.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/9EB84598-972B-DE11-95BE-000423D98B5C.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/AA46709E-E02B-DE11-A224-000423D9870C.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/B632F749-982B-DE11-8521-001D09F23A20.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/B84249BD-982B-DE11-A270-0019DB29C5FC.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/C0DF59B6-9E2B-DE11-99E2-000423D94E70.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/D267D7D2-E72B-DE11-8EB2-000423D98DC4.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/DC4B5D45-D42B-DE11-9F9E-000423D94990.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/E431A092-9C2B-DE11-9CFE-000423D99EEE.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/E65E9798-E32B-DE11-9C8D-001617E30D0A.root',
       '/store/relval/CMSSW_3_1_0_pre5/RelValQCD_Pt_3000_3500/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0000/FAC90682-982B-DE11-8865-001617C3B6C6.root'

    ] ),
                            skipEvents = cms.untracked.uint32(1486)
                            )

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
                                        ignoreTotal = cms.untracked.int32(-1)
                                        )

#process.out = cms.OutputModule("PoolOutputModule",
#                               outputCommands = cms.untracked.vstring('keep *'),
#                               fileName = cms.untracked.string('/tmp/tropiano/tracks_100_ann80-9-4-1-1-1(prova).root')
#                               )
from RecoTracker.FinalTrackSelectors.TracksWithQuality_cff import *
#process.withLooseQuality.src = 'ctfWithMaterialTracksDAF'
process.ctfWithMaterialTracksDAF.TrajectoryInEvent = True


process.validation = cms.Sequence(process.multiTrackValidator)

process.p = cms.Path(process.TrackRefitter*process.ctfWithMaterialTracksDAF*process.validation)
#process.p = cms.Path(process.mix*process.TrackRefitter*process.ctfWithMaterialTracksDAF)

#process.outpath = cms.EndPath(process.out)
process.ctfWithMaterialTracksDAF.src = 'TrackRefitter'


process.MRHFittingSmoother.EstimateCut = -1
process.MRHFittingSmoother.MinNumberOfHits = 3
process.multiTrackValidator.outputFile = 'prova_grouped.root'
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

