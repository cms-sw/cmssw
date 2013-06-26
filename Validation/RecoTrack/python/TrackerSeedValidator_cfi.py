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
    
    # set true if you do not want efficiency fakes and resolution fit
    # to be calculated in the end run (for automated validation):
    skipHistoFit=cms.untracked.bool(False),

    runStandalone = cms.bool(True),
    
    ### matching configuration ###
    associators = cms.vstring('TrackAssociatorByHits'),    

    ### sim input configuration ###
    label_tp_effic = cms.InputTag("mix","MergedTrackTruth"),
    label_tp_fake = cms.InputTag("mix","MergedTrackTruth"),
    label_tv = cms.InputTag("mix","MergedTrackTruth"),
    label_pileupinfo = cms.InputTag("addPileupInfo"),
    sim = cms.string('g4SimHits'),
    parametersDefiner = cms.string('LhcParametersDefinerForTP'),          # collision like tracks

    ### reco input configuration ###
    label = cms.VInputTag(cms.InputTag("initialStepSeeds")),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    
    ### output configuration
    dirName = cms.string('Tracking/Seed/'),
    outputFile = cms.string(''),

    TTRHBuilder = cms.string('WithTrackAngle')
)


