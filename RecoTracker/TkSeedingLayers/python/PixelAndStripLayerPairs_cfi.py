import FWCore.ParameterSet.Config as cms

# Seeding with one hit in outer pixel and one in inner strip.
# Useful for exotic physics, V0 finding etc.

PixelAndStripLayerPairs = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring(
        'BPix3+TIB1_pos',
        'BPix2+TIB1_pos',
        'BPix3+TIB2_pos',
        'FPix1+TIB1_pos',
#        'FPix1_pos+TID1_pos',
#        'FPix2+TIB1_pos',
        'FPix2_pos+TID1_pos',
        'FPix2_pos+TID2_pos',
        'FPix2_pos+TID3_pos',
        'FPix2_pos+TEC1_pos',
        'BPix3+TIB1_neg',
        'BPix2+TIB1_neg',
        'BPix3+TIB2_neg',
        'FPix1+TIB1_neg',
#        'FPix1_neg+TID1_neg',
#        'FPix2+TIB1_neg',
        'FPix2_neg+TID1_neg',
        'FPix2_neg+TID2_neg',
        'FPix2_neg+TID3_neg',
        'FPix2_neg+TEC1_neg'),
                                 
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TID = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(1)
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(1)
    ),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('siPixelRecHits'),
    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('siPixelRecHits'),
    )
)


