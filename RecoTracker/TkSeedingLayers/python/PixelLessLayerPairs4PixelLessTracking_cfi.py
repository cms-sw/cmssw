import FWCore.ParameterSet.Config as cms

### standard configuration of *strip* layer pairs to be used 
### to reconstruct tracks without using additional pixel-with tracking steps. 

from RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi import *

pixelLessLayerPairs4PixelLessTracking = seedingLayersEDProducer.clone()
pixelLessLayerPairs4PixelLessTracking.layerList = cms.vstring(
        'TIB1+TIB2','TIB1+TIB3','TIB2+TIB3',
        'TIB1+TID1_pos', 'TIB1+TID1_neg',
        'TIB2+TID1_pos', 'TIB2+TID1_neg',
        'TIB1+TID2_pos', 'TIB1+TID2_neg',
        'TID1_pos+TID2_pos', 
        'TID2_pos+TID3_pos', 
        'TID3_pos+TEC2_pos', 
        'TEC1_pos+TEC2_pos', 
        'TEC2_pos+TEC3_pos', 
        'TID1_neg+TID2_neg', 
        'TID2_neg+TID3_neg', 
        'TID3_neg+TEC2_neg', 
        'TEC1_neg+TEC2_neg',
        'TEC2_neg+TEC3_neg'
)
pixelLessLayerPairs4PixelLessTracking.TID1 = cms.PSet(
    useSimpleRphiHitsCleaner = cms.bool(False),
    minRing = cms.int32(1),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched"),
    stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
    useRingSlector = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    maxRing = cms.int32(3)
)
pixelLessLayerPairs4PixelLessTracking.TID2 = cms.PSet(
    useSimpleRphiHitsCleaner = cms.bool(False),
    minRing = cms.int32(1),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched"),
    stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
    useRingSlector = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    maxRing = cms.int32(3)
)
pixelLessLayerPairs4PixelLessTracking.TID3 = cms.PSet(
    useSimpleRphiHitsCleaner = cms.bool(False),        
    minRing = cms.int32(1),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched"),
    stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
    useRingSlector = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    maxRing = cms.int32(2)
)

pixelLessLayerPairs4PixelLessTracking.TEC = cms.PSet(
    useSimpleRphiHitsCleaner = cms.bool(False),        
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched"),
    stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
    useRingSlector = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    minRing = cms.int32(1),
    maxRing = cms.int32(2)
)
pixelLessLayerPairs4PixelLessTracking.TIB1 = cms.PSet(
    TTRHBuilder = cms.string('WithTrackAngle'),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched"),
    stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
    useSimpleRphiHitsCleaner = cms.bool(False)
)
pixelLessLayerPairs4PixelLessTracking.TIB2 = cms.PSet(
    TTRHBuilder = cms.string('WithTrackAngle'),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched"),
    stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
    useSimpleRphiHitsCleaner = cms.bool(False)
)
pixelLessLayerPairs4PixelLessTracking.TIB3 = cms.PSet(
    TTRHBuilder = cms.string('WithTrackAngle'),
    rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHitUnmatched"),
    stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHitUnmatched"),
    useSimpleRphiHitsCleaner = cms.bool(False)
)



