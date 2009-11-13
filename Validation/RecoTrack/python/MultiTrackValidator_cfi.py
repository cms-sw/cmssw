import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.TrackingParticleSelectionForEfficiency_cfi import *
from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from SimTracker.TrackAssociation.CosmicParametersDefinerForTP_cfi import *

multiTrackValidator = cms.EDFilter("MultiTrackValidator",
    # selection of TP for evaluation of efficiency
    TrackingParticleSelectionForEfficiency,
    useFabsEta = cms.bool(False),
    associatormap = cms.InputTag("trackingParticleRecoTrackAsssociation"),
    #associatormap = cms.InputTag("assoc2secStepTk"),
    #associatormap = cms.InputTag("assoc2thStepTk"),
    #associatormap = cms.InputTag("assoc2GsfTracks"),
    label_tp_effic = cms.InputTag("mergedtruth","MergedTrackTruth"),
    label_tp_fake = cms.InputTag("mergedtruth","MergedTrackTruth"),
    label = cms.VInputTag(cms.InputTag("generalTracks")),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    sim = cms.string('g4SimHits'),
    outputFile = cms.string(''),
    # collision like tracks
    parametersDefiner = cms.string('LhcParametersDefinerForTP'),
    # cosmics tracks
    # parametersDefiner = cms.string('CosmicParametersDefinerForTP'), 
    # set true if you do not want that MTV launch an exception
    # if the track collectio is missing (e.g. HLT):
    ignoremissingtrackcollection=cms.untracked.bool(False),
    #
    # set true if you do not want efficiency fakes and resolution fit
    # to be calculated in the end run (for automated validation):
    skipHistoFit=cms.untracked.bool(True),
    associators = cms.vstring('TrackAssociatorByHitsRecoDenom'),
    useInvPt = cms.bool(False),
    dirName = cms.string('Tracking/Track/'),
    #
    min = cms.double(-2.5),
    max = cms.double(2.5),
    nint = cms.int32(50),
    #
    ptRes_rangeMin = cms.double(-0.1),
    ptRes_rangeMax = cms.double(0.1),
    phiRes_rangeMin = cms.double(-0.003),
    phiRes_rangeMax = cms.double(0.003),
    cotThetaRes_rangeMin = cms.double(-0.01),
    cotThetaRes_rangeMax = cms.double(+0.01),
    dxyRes_rangeMin = cms.double(-0.01),
    dxyRes_rangeMax = cms.double(0.01),
    dzRes_rangeMin = cms.double(-0.05),
    dzRes_rangeMax = cms.double(+0.05),
    # 
    ptRes_nbin = cms.int32(100),                                   
    phiRes_nbin = cms.int32(100),                                   
    cotThetaRes_nbin = cms.int32(120),                                   
    dxyRes_nbin = cms.int32(100),                                   
    dzRes_nbin = cms.int32(150),                                   
    # 
    minpT = cms.double(0.1),
    maxpT = cms.double(1000),
    nintpT = cms.int32(40),
    #                               
    minHit = cms.double(-0.5),                            
    maxHit = cms.double(34.5),
    nintHit = cms.int32(35),
    #
    minPhi = cms.double(-3.15),
    maxPhi = cms.double(3.15),
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
    nintZpos = cms.int32(100),                               
    # if *not* uses associators, the TP-RecoTrack maps has to be specified 
    UseAssociators = cms.bool(True),
    useLogPt=cms.untracked.bool(False),
    useGsf=cms.bool(False)
)
