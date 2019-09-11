import FWCore.ParameterSet.Config as cms

from RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi import *

SiStripLayerTriplets = seedingLayersEDProducer.clone()
SiStripLayerTriplets.layerList = cms.vstring(
	#for barrel SiStrip seeding
	'TIB1+TIB2+TIB3',
	'TIB2+TIB3+TIB4',
	'TIB3+TIB4+TOB1',
	'TIB4+TOB1+TOB2',
	'TOB1+TOB2+TOB3',
	'TOB2+TOB3+TOB4',
	'TOB3+TOB4+TOB5',
	'TOB4+TOB5+TOB6',

	#for endcap SiStrip seeding
	'TID1_pos+TID2_pos+TID3_pos',
	'TID2_pos+TID3_pos+TEC1_pos',
	'TID3_pos+TEC1_pos+TEC2_pos',
	'TEC1_pos+TEC2_pos+TEC3_pos',
	'TEC2_pos+TEC3_pos+TEC4_pos',
	'TEC4_pos+TEC5_pos+TEC6_pos',
	'TEC6_pos+TEC7_pos+TEC8_pos',
	'TEC7_pos+TEC8_pos+TEC9_pos',

	'TID1_neg+TID2_neg+TID3_neg',
	'TID2_neg+TID3_neg+TEC1_neg',
	'TID3_neg+TEC1_neg+TEC2_neg',
	'TEC1_neg+TEC2_neg+TEC3_neg',
	'TEC2_neg+TEC3_neg+TEC4_neg',
	'TEC4_neg+TEC5_neg+TEC6_neg',
	'TEC6_neg+TEC7_neg+TEC8_neg',
	'TEC7_neg+TEC8_neg+TEC9_neg',


	#mixed barel and endcap SiStrip seeding


	'TIB1+TID1_pos+TID2_pos',
	'TIB2+TID1_pos+TID2_pos',
	'TIB3+TID1_pos+TID2_pos',
	'TIB4+TID1_pos+TID2_pos',

	'TID2_pos+TID3_pos+TEC1_pos',
	'TID3_pos+TEC1_pos+TEC2_pos',

	'TOB1+TEC1_pos+TEC2_pos',
	'TOB2+TEC1_pos+TEC2_pos',
	'TOB3+TEC1_pos+TEC2_pos',
	'TOB4+TEC1_pos+TEC2_pos',
	'TOB5+TEC1_pos+TEC2_pos',
	'TOB6+TEC1_pos+TEC2_pos',

	'TIB1+TID1_neg+TID2_neg',
	'TIB2+TID1_neg+TID2_neg',
	'TIB3+TID1_neg+TID2_neg',
	'TIB4+TID1_neg+TID2_neg',

	'TID2_neg+TID3_neg+TEC1_neg',
	'TID3_neg+TEC1_neg+TEC2_neg',

	'TOB1+TEC1_neg+TEC2_neg',
	'TOB2+TEC1_neg+TEC2_neg',
	'TOB3+TEC1_neg+TEC2_neg',
	'TOB4+TEC1_neg+TEC2_neg',
	'TOB5+TEC1_neg+TEC2_neg',
	'TOB6+TEC1_neg+TEC2_neg'
)




SiStripLayerTriplets.TOB = cms.PSet(
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    TTRHBuilder = cms.string('WithTrackAngle')
    ,clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone'))
)

SiStripLayerTriplets.TIB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
        ,clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone'))
)

SiStripLayerTriplets.TID = cms.PSet(
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    useRingSlector = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    minRing = cms.int32(1),
    maxRing = cms.int32(2)
   ,clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone'))
)

SiStripLayerTriplets.TEC = cms.PSet(
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    useRingSlector = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    minRing = cms.int32(1),
    maxRing = cms.int32(2)
   ,clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone'))
)
