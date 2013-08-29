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
    associatormap = cms.InputTag("trackingParticleRecoTrackAsssociation"),
    #associatormap = cms.InputTag("assoc2secStepTk"),
    #associatormap = cms.InputTag("assoc2thStepTk"),
    #associatormap = cms.InputTag("assoc2GsfTracks"),
    associators = cms.vstring('TrackAssociatorByHitsRecoDenom'),    
    UseAssociators = cms.bool(True), # if False, the TP-RecoTrack maps has to be specified 

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
)
