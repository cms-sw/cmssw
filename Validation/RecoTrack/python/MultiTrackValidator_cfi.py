import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.TrackingParticleSelectionForEfficiency_cfi import *
multiTrackValidator = cms.EDFilter("MultiTrackValidator",
    #InputTag associatormap = assoc2secStepTk
    #InputTag associatormap = assoc2thStepTk   
    # selection of TP for evaluation of efficiency
    TrackingParticleSelectionForEfficiency,
    useFabsEta = cms.bool(False),
    minpT = cms.double(0.0),
    nintHit = cms.int32(35),
    associatormap = cms.InputTag("trackingParticleRecoTrackAsssociation"),
    label_tp_fake = cms.InputTag("mergedtruth","MergedTrackTruth"),
    out = cms.string('validationPlots.root'),
    min = cms.double(-2.5),
    nintpT = cms.int32(200),
    label = cms.VInputTag(cms.InputTag("generalTracks")),
    maxHit = cms.double(35.0),
    label_tp_effic = cms.InputTag("mergedtruth","MergedTrackTruth"),
    useInvPt = cms.bool(False),
    dirName = cms.string('RecoTrackV/Track/'),
    minHit = cms.double(0.0),
    # 
    sim = cms.string('g4SimHits'),
    # 
    associators = cms.vstring('TrackAssociatorByHits', 
        'TrackAssociatorByChi2'),
    max = cms.double(2.5),
    maxpT = cms.double(100.0),
    nint = cms.int32(50),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    # if *not* uses associators, the TP-RecoTrack maps has to be specified 
    UseAssociators = cms.bool(False)
)


