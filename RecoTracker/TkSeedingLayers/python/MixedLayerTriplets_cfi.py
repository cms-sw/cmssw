import FWCore.ParameterSet.Config as cms

mixedlayertriplets = cms.ESProducer("MixedLayerTripletsESProducer",
    layerList = cms.vstring('BPix1+BPix2+BPix3', 'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg', 'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg', 'BPix1+BPix2+TIB1', 'BPix1+BPix3+TIB1', 'BPix2+BPix3+TIB1'),
    ComponentName = cms.string('MixedLayerTriplets'),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('siPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    ),
    TID = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedTriplets'),
        HitProducer = cms.string('siPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)


