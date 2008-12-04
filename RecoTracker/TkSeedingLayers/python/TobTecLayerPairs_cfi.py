# The following comments couldn't be translated into the new config version:

#		       "TEC1_pos+TEC2_pos", "TEC2_pos+TEC3_pos", "TEC3_pos+TEC4_pos", "TEC4_pos+TEC5_pos", "TEC5_pos+TEC6_pos", "TEC6_pos+TEC7_pos",

import FWCore.ParameterSet.Config as cms

tobteclayerpairs = cms.ESProducer("TobTecLayerPairsESProducer",
    layerList = cms.vstring('TOB1+TOB2', 
        'TOB1+TEC1_pos', 
        'TOB1+TEC2_pos', 
        'TOB1+TEC1_neg', 
        'TOB1+TEC2_neg', 
        'TEC1_pos+TEC2_pos', 
        'TEC1_pos+TEC3_pos', 
        'TEC2_pos+TEC3_pos', 
        'TEC2_pos+TEC4_pos', 
        'TEC3_pos+TEC4_pos', 
        'TEC3_pos+TEC5_pos', 
        'TEC4_pos+TEC5_pos', 
        'TEC4_pos+TEC6_pos', 
        'TEC5_pos+TEC6_pos', 
        'TEC5_pos+TEC7_pos', 
        'TEC6_pos+TEC7_pos', 
        'TEC1_neg+TEC2_neg', 
        'TEC1_neg+TEC3_neg', 
        'TEC2_neg+TEC3_neg', 
        'TEC2_neg+TEC4_neg', 
        'TEC3_neg+TEC4_neg', 
        'TEC3_neg+TEC5_neg', 
        'TEC4_neg+TEC5_neg', 
        'TEC4_neg+TEC6_neg', 
        'TEC5_neg+TEC6_neg', 
        'TEC5_neg+TEC7_neg', 
        'TEC6_neg+TEC7_neg'),
    # We use ring 5 and 6 of TEC disks 1 and 2, so these override the TEC
    #   section above.  Note that TEC ring 6 uses rPhi hits only, so we need
    #   to include both hit types for this region.
    TEC1_neg = cms.PSet(
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(6)
    ),
    ComponentName = cms.string('TobTecLayerPairs'),
    TEC1_pos = cms.PSet(
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(6)
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(5),
        maxRing = cms.int32(5)
    ),
    TEC2_neg = cms.PSet(
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(6)
    ),
    TOB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TEC2_pos = cms.PSet(
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(6)
    )
)


