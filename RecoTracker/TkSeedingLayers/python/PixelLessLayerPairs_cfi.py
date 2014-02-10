import FWCore.ParameterSet.Config as cms

PixelLessLayerPairs = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('TIB1+TIB2', 
        'TIB1+TID1_pos', 
#        'TIB1+TID2_pos', 
        'TIB1+TID1_neg', 
#        'TIB1+TID2_neg', 
        'TID1_pos+TID2_pos', 
        'TID2_pos+TID3_pos', 
        'TID3_pos+TEC1_pos', 
        'TEC1_pos+TEC2_pos', 
        'TEC2_pos+TEC3_pos', 
        'TEC3_pos+TEC4_pos',
        'TEC3_pos+TEC5_pos', 
        'TEC4_pos+TEC5_pos',                             
        'TID1_neg+TID2_neg', 
        'TID2_neg+TID3_neg', 
        'TID3_neg+TEC1_neg', 
        'TEC1_neg+TEC2_neg', 
        'TEC2_neg+TEC3_neg',
        'TEC3_neg+TEC4_neg',
        'TEC3_neg+TEC5_neg',
        'TEC4_neg+TEC5_neg'),

# WARNING: in the old implemenation, all the 3 rings of  TID were used.
# we need a different configuaration of rings for TID disks. Is it feasible 
# in the current framework?? 

    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TID = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    )
)


