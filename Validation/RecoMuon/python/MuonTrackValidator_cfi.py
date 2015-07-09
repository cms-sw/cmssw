import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from SimTracker.TrackAssociation.CosmicParametersDefinerForTP_cfi import *

muonTrackValidator = cms.EDAnalyzer("MuonTrackValidator",
    # input TrackingParticle collections
    label_tp_effic = cms.InputTag("mix","MergedTrackTruth"),
    label_tp_fake = cms.InputTag("mix","MergedTrackTruth"),
    # input reco::Track collection
    label = cms.VInputTag(cms.InputTag("globalMuons")),
    # switches to be set according to the input Track collection to properly count SimHits
    usetracker = cms.bool(True),
    usemuon = cms.bool(True),
    #
    useGsf=cms.bool(False),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    # set true if you do not want that MTV launch an exception
    # if the track collection is missing (e.g. HLT):
    ignoremissingtrackcollection=cms.untracked.bool(False),
    #
    # selection of TP for evaluation of efficiency, from "TrackingParticleSelectionForEfficiency"
    signalOnlyTP = cms.bool(True),
    intimeOnlyTP = cms.bool(False),
    stableOnlyTP = cms.bool(False),
    chargedOnlyTP = cms.bool(True),
    pdgIdTP = cms.vint32(13,-13),
    minHitTP = cms.int32(0),
    ptMinTP = cms.double(0.9),
    minRapidityTP = cms.double(-2.4),
    maxRapidityTP = cms.double(2.4),
    tipTP = cms.double(3.5),
    lipTP = cms.double(30.0),
    # collision-like tracks
    parametersDefiner = cms.string('LhcParametersDefinerForTP'),
    # cosmics tracks
    # parametersDefiner = cms.string('CosmicParametersDefinerForTP'), 
    #
    # map linking SimHits to TrackingParticles, needed for cosmics validation`
    simHitTpMapTag = cms.InputTag("simHitTPAssocProducer"), 
    #
    # if *not* uses associators, the TP-RecoTrack maps has to be specified 
    UseAssociators = cms.bool(False),
    associators = cms.vstring('a_MuonAssociator'),
    associatormap = cms.InputTag("tpToMuonTrackAssociation"),
    #
    # BiDirectional Logic for RecoToSim association corrects the Fake rates (counting ghosts and split tracks as fakes)
    #  setting it to False the ghost and split tracks are counted as good ones (old setting of Muon Validation up to CMSSW_3_6_0_pre4)
    #  the default setting is True: should NOT be changed !
    BiDirectional_RecoToSim_association = cms.bool(True),
    #
    # Output File / Directory
    outputFile = cms.string(''),           
    dirName = cms.string('Muons/RecoMuonV/MultiTrack/'),
    #
    # Parameters for plots                                    
    useFabsEta = cms.bool(False),
    min = cms.double(-2.5),
    max = cms.double(2.5),
    nint = cms.int32(50),
    #
    ptRes_nbin = cms.int32(100),                                   
    ptRes_rangeMin = cms.double(-0.3),
    ptRes_rangeMax = cms.double(0.3),
    #
    phiRes_nbin = cms.int32(100),                                   
    phiRes_rangeMin = cms.double(-0.05),
    phiRes_rangeMax = cms.double(0.05),
    #
    etaRes_rangeMin = cms.double(-0.05),
    etaRes_rangeMax = cms.double(0.05),
    #
    cotThetaRes_nbin = cms.int32(120),                                   
    cotThetaRes_rangeMin = cms.double(-0.01),
    cotThetaRes_rangeMax = cms.double(0.01),
    #
    dxyRes_nbin = cms.int32(100),                                   
    dxyRes_rangeMin = cms.double(-0.02),
    dxyRes_rangeMax = cms.double(0.02),
    #
    dzRes_nbin = cms.int32(150),                                   
    dzRes_rangeMin = cms.double(-0.05),
    dzRes_rangeMax = cms.double(0.05),
    # 
    minpT = cms.double(0.1),
    maxpT = cms.double(1500),
    nintpT = cms.int32(40),
    useLogPt=cms.untracked.bool(False),
    useInvPt = cms.bool(False),
    #                               
    minHit = cms.double(-0.5),                            
    maxHit = cms.double(74.5),
    nintHit = cms.int32(75),
    #
    minPhi = cms.double(-3.1416),
    maxPhi = cms.double(3.1416),
    nintPhi = cms.int32(36),
    #
    minDxy = cms.double(-3),
    maxDxy = cms.double(3),
    nintDxy = cms.int32(100),
    #
    minDz = cms.double(-10),
    maxDz = cms.double(10),
    nintDz = cms.int32(100),
    # TP originating vertical position
    minVertpos = cms.double(0),
    maxVertpos = cms.double(5),
    nintVertpos = cms.int32(100),
    # TP originating z position
    minZpos = cms.double(-10),
    maxZpos = cms.double(10),
    nintZpos = cms.int32(100)
)
