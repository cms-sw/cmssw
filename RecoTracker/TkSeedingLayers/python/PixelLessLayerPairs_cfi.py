import FWCore.ParameterSet.Config as cms

pixellesslayerpairs = cms.ESProducer("PixelLessLayerPairsESProducer",
    layerList = cms.vstring('TIB1+TIB2', 'TIB1+TID1_pos', 'TIB1+TID2_pos', 'TIB1+TID1_neg', 'TIB1+TID2_neg', 'TID1_pos+TID2_pos', 'TID2_pos+TID3_pos', 'TID3_pos+TEC2_pos', 'TEC2_pos+TEC3_pos', 'TID1_neg+TID2_neg', 'TID2_neg+TID3_neg', 'TID3_neg+TEC2_neg', 'TEC2_neg+TEC3_neg'),
    TID1 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(2)
    ),
    TID3 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(2)
    ),
    TID2 = cms.PSet(
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        minRing = cms.int32(1),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(3)
    ),
    ComponentName = cms.string('PixelLessLayerPairs'),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    #WARNING: in the old implemenation, all the 3 rings of  TID were used.
    # we need a different configuaration of rings for TID disks. Is it feasible 
    # in the current framework?? 
    TIB = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    )
)


