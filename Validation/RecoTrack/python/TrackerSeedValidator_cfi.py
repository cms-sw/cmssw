import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.TrackingParticleSelectionForEfficiency_cfi import *
from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from Validation.RecoTrack.MTVHistoProducerAlgoForTrackerBlock_cfi import *

trackerSeedValidator = cms.EDAnalyzer("TrackerSeedValidator",

    ### general settings ###
    # selection of TP for evaluation of efficiency #
    TrackingParticleSelectionForEfficiency,
    
    # HistoProducerAlgo. Defines the set of plots to be booked and filled
    histoProducerAlgoBlock = MTVHistoProducerAlgoForTrackerBlock,

    # set true if you do not want that MTV launch an exception
    # if the track collectio is missing (e.g. HLT):
    ignoremissingtrackcollection=cms.untracked.bool(False),
    
    ### matching configuration ###
    associators = cms.VInputTag("trackAssociatorByHits"),

    ### sim input configuration ###
    label_tp_effic = cms.InputTag("mix","MergedTrackTruth"),
    label_tp_fake = cms.InputTag("mix","MergedTrackTruth"),
    label_pileupinfo = cms.InputTag("addPileupInfo"),
    sim = cms.VInputTag(
      cms.InputTag("g4SimHits", "TrackerHitsPixelBarrelHighTof"),
      cms.InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"),
      cms.InputTag("g4SimHits", "TrackerHitsPixelEndcapHighTof"),
      cms.InputTag("g4SimHits", "TrackerHitsPixelEndcapLowTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTECHighTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTECLowTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTIBHighTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTIBLowTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTIDHighTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTIDLowTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTOBHighTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTOBLowTof")
    ),
    parametersDefiner = cms.string('LhcParametersDefinerForTP'),          # collision like tracks

    ### reco input configuration ###
    label = cms.VInputTag(cms.InputTag("initialStepSeeds")),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    
    ### output configuration
    dirName = cms.string('Tracking/Seed/'),

    TTRHBuilder = cms.string('WithTrackAngle')
)


