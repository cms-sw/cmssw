import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.TrackingParticleSelectionForEfficiency_cfi import *
from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from SimTracker.TrackAssociation.CosmicParametersDefinerForTP_cfi import *
from Validation.RecoTrack.MTVHistoProducerAlgoForTrackerBlock_cfi import *

multiTrackValidator = cms.EDAnalyzer(
    "MultiTrackValidator",

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
    skipHistoFit=cms.untracked.bool(True),

    runStandalone = cms.bool(False),

    useGsf=cms.bool(False),

    
    ### matching configuration ###
    # Example of TP-Track map
    associators = cms.untracked.VInputTag("trackingParticleRecoTrackAsssociation"),
    # Example of associator
    #associators = cms.untracked.VInputTag("quickTrackAssociatorByHits"),
    # if False, the src's above should specify the TP-RecoTrack association
    # if True, the src's above should specify the associator
    UseAssociators = cms.bool(False),

    ### sim input configuration ###
    label_tp_effic = cms.InputTag("mix","MergedTrackTruth"),
    label_tp_fake = cms.InputTag("mix","MergedTrackTruth"),
    label_tv = cms.InputTag("mix","MergedTrackTruth"),
    label_pileupinfo = cms.InputTag("addPileupInfo"),
    sim = cms.string('g4SimHits'),
    parametersDefiner = cms.string('LhcParametersDefinerForTP'),          # collision like tracks
    # parametersDefiner = cms.string('CosmicParametersDefinerForTP'),     # cosmics tracks
    simHitTpMapTag = cms.InputTag("simHitTPAssocProducer"),               # needed by CosmicParametersDefinerForTP

    ### reco input configuration ###
    label = cms.VInputTag(cms.InputTag("generalTracks")),
    beamSpot = cms.InputTag("offlineBeamSpot"),

    ### dE/dx configuration ###
    dEdx1Tag = cms.InputTag("dedxHarmonic2"),
    dEdx2Tag = cms.InputTag("dedxTruncated40"),
    
    ### output configuration
    dirName = cms.string('Tracking/Track/'),
    outputFile = cms.string(''),

    ### for fake rate vs dR ###
    trackCollectionForDrCalculation = cms.InputTag("generalTracks")
)
