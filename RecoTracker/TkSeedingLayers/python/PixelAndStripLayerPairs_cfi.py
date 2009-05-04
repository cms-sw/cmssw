import FWCore.ParameterSet.Config as cms

# Seeding with one hit in outer pixel and one in inner strip.
# Useful for exotic physics, V0 finding etc.

pixelandstriplayerpairs = cms.ESProducer("SeedingLayersESProducer",
    ComponentName = cms.string('PixelAndStripLayerPairs'),
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
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('siPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('siPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    )
)


