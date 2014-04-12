import FWCore.ParameterSet.Config as cms

TobTecLayerPairs = cms.EDProducer("SeedingLayersEDProducer",
# Don't bother with TEC8 and 9, as tracking requires 2 hits outside
# the seeding pairs.
    layerList = cms.vstring('TOB1+TOB2', 
        'TOB1+TEC1_pos', 
#        'TOB1+TEC2_pos', 
        'TOB1+TEC1_neg', 
#        'TOB1+TEC2_neg', 
        'TEC1_pos+TEC2_pos', 
#        'TEC1_pos+TEC3_pos', 
        'TEC2_pos+TEC3_pos', 
#        'TEC2_pos+TEC4_pos', 
        'TEC3_pos+TEC4_pos', 
#        'TEC3_pos+TEC5_pos', 
        'TEC4_pos+TEC5_pos', 
#        'TEC4_pos+TEC6_pos', 
        'TEC5_pos+TEC6_pos', 
#        'TEC5_pos+TEC7_pos', 
        'TEC6_pos+TEC7_pos', 
        'TEC1_neg+TEC2_neg', 
#        'TEC1_neg+TEC3_neg', 
        'TEC2_neg+TEC3_neg', 
#        'TEC2_neg+TEC4_neg', 
        'TEC3_neg+TEC4_neg', 
#        'TEC3_neg+TEC5_neg', 
        'TEC4_neg+TEC5_neg', 
#        'TEC4_neg+TEC6_neg', 
        'TEC5_neg+TEC6_neg', 
#        'TEC5_neg+TEC7_neg', 
        'TEC6_neg+TEC7_neg'),

    TOB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),

    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(5),
        maxRing = cms.int32(5)
    )
)


