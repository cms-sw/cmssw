import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.TrackingParticleSelectionForEfficiency_cfi import *
from Validation.RecoTrack.GenParticleSelectionsForEfficiency_cff import *
from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from SimTracker.TrackAssociation.CosmicParametersDefinerForTP_cfi import *
from Validation.RecoTrack.MTVHistoProducerAlgoForTrackerBlock_cfi import *

multiTrackValidatorGenPs = cms.EDAnalyzer(
    "MultiTrackValidatorGenPs",

    ### general settings ###
    #ok this is not used, but is needed for the MTV contructor
    TrackingParticleSelectionForEfficiency,
    # selection of GP for evaluation of efficiency #
    GenParticleSelectionForEfficiency,
    
    # HistoProducerAlgo. Defines the set of plots to be booked and filled
    histoProducerAlgoBlock = MTVHistoProducerAlgoForTrackerBlock,

    # set true if you do not want that MTV launch an exception
    # if the track collectio is missing (e.g. HLT):
    ignoremissingtrackcollection=cms.untracked.bool(False),
    
    useGsf=cms.bool(False),

    
    ### matching configuration ###
    associators = cms.untracked.VInputTag("TrackAssociatorByChi2"),
    UseAssociators = cms.bool(True), # if False, the TP-RecoTrack maps has to be specified 

    ### sim input configuration ###
    label_tp_effic = cms.InputTag("genParticles"),
    label_tp_fake = cms.InputTag("genParticles"),
    label_tv = cms.InputTag("mix","MergedTrackTruth"),#this is not used
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
    # parametersDefiner = cms.string('CosmicParametersDefinerForTP'),     # cosmics tracks

    ### reco input configuration ###
    label = cms.VInputTag(cms.InputTag("generalTracks")),
    beamSpot = cms.InputTag("offlineBeamSpot"),

    ### dE/dx configuration ###
    dEdx1Tag = cms.InputTag("dedxHarmonic2"),
    dEdx2Tag = cms.InputTag("dedxTruncated40"),
    
    ### output configuration
    dirName = cms.string('Tracking/Track/'),

    ### Allow switching off particular histograms
    doSimPlots = cms.untracked.bool(True),
    doSimTrackPlots = cms.untracked.bool(True),
    doRecoTrackPlots = cms.untracked.bool(True),
    dodEdxPlots = cms.untracked.bool(False),
)
