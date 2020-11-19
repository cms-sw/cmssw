import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.TrackingParticleSelectionForEfficiency_cfi import *
from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from SimTracker.TrackAssociation.CosmicParametersDefinerForTP_cfi import *
from Validation.RecoTrack.MTVHistoProducerAlgoForTrackerBlock_cfi import *

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
multiTrackValidator = DQMEDAnalyzer(
    "MultiTrackValidator",

    ### general settings ###
    # selection of TP for evaluation of efficiency #
    TrackingParticleSelectionForEfficiency,

    # HistoProducerAlgo. Defines the set of plots to be booked and filled
    histoProducerAlgoBlock = MTVHistoProducerAlgoForTrackerBlock,

    # set true if you do not want that MTV launch an exception
    # if the track collectio is missing (e.g. HLT):
    ignoremissingtrackcollection=cms.untracked.bool(False),

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
    label_tp_effic_refvector = cms.bool(False),
    label_tp_fake_refvector = cms.bool(False),
    label_tv = cms.InputTag("mix","MergedTrackTruth"),
    label_pileupinfo = cms.InputTag("addPileupInfo"),
    sim = cms.VInputTag(
      cms.InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"),
      cms.InputTag("g4SimHits", "TrackerHitsPixelBarrelHighTof"),
      cms.InputTag("g4SimHits", "TrackerHitsPixelEndcapLowTof"),
      cms.InputTag("g4SimHits", "TrackerHitsPixelEndcapHighTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTIBLowTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTIBHighTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTIDLowTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTIDHighTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTOBLowTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTOBHighTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTECLowTof"),
      cms.InputTag("g4SimHits", "TrackerHitsTECHighTof"),
    ),
    parametersDefiner = cms.string('LhcParametersDefinerForTP'),          # collision like tracks
    # parametersDefiner = cms.string('CosmicParametersDefinerForTP'),     # cosmics tracks
    simHitTpMapTag = cms.InputTag("simHitTPAssocProducer"),               # needed by CosmicParametersDefinerForTP

    label_tp_nlayers = cms.InputTag("trackingParticleNumberOfLayersProducer", "trackerLayers"),
    label_tp_npixellayers = cms.InputTag("trackingParticleNumberOfLayersProducer", "pixelLayers"),
    label_tp_nstripstereolayers = cms.InputTag("trackingParticleNumberOfLayersProducer", "stripStereoLayers"),

    ### reco input configuration ###
    label = cms.VInputTag(cms.InputTag("generalTracks")),
    beamSpot = cms.InputTag("offlineBeamSpot"),

    ### selection MVA
    mvaLabels = cms.untracked.PSet(),

    ### dE/dx configuration ###
    dEdx1Tag = cms.InputTag("dedxHarmonic2"),
    dEdx2Tag = cms.InputTag("dedxTruncated40"),

    ### output configuration
    dirName = cms.string('Tracking/Track/'),

    ### for fake rate vs dR ###
    # True=use collection below; False=use "label" collection itself
    calculateDrSingleCollection = cms.untracked.bool(True),
    trackCollectionForDrCalculation = cms.InputTag("generalTracks"),

    ### Do plots only if first reco vertex is from hard scatter?
    doPlotsOnlyForTruePV = cms.untracked.bool(False),
    label_vertex = cms.untracked.InputTag("offlinePrimaryVertices"),
    vertexAssociator = cms.untracked.InputTag("VertexAssociatorByPositionAndTracks"),

    simPVMaxZ = cms.untracked.double(-1),

    ### Allow switching off particular histograms
    doSummaryPlots = cms.untracked.bool(True),
    doSimPlots = cms.untracked.bool(True),
    doSimTrackPlots = cms.untracked.bool(True),
    doRecoTrackPlots = cms.untracked.bool(True),
    dodEdxPlots = cms.untracked.bool(False),
    doPVAssociationPlots = cms.untracked.bool(False), # do plots that require true PV, if True, label_vertex and vertexAssociator are read
    doSeedPlots = cms.untracked.bool(False), # input comes from TrackFromSeedProducer    
    doMVAPlots = cms.untracked.bool(False), # needs input from track MVA selectors

    ### do resolution plots only for these labels (or all if empty)
    doResolutionPlotsForLabels = cms.VInputTag(),

    cores = cms.InputTag("highPtJetsForTrk"), #ak4CaloJets with pt>1 TeV
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(multiTrackValidator, sim = ['fastSimProducer:TrackerHits'])

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(multiTrackValidator,
    label_tp_effic = "mixData:MergedTrackTruth",
    label_tp_fake = "mixData:MergedTrackTruth",
    label_tv = "mixData:MergedTrackTruth",
)
