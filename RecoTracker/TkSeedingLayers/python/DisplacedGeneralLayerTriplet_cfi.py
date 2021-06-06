import FWCore.ParameterSet.Config as cms

from RecoTracker.TkSeedingLayers.seedingLayersEDProducer_cfi import *

DisplacedGeneralLayerTriplet = seedingLayersEDProducer.clone(
        layerList = cms.vstring( 
                #----------
                #TIB
                #----------

                'TIB1+TIB2+MTIB3',
                'TIB1+TIB2+MTIB4',
                'TIB1+MTIB3+MTIB4',
                'TIB2+MTIB3+MTIB4',

                #----------
                #TOB
                #----------
                'TOB1+TOB2+MTOB3',
                'TOB2+MTOB3+MTOB4',
                'MTOB3+MTOB4+MTOB5',
                'MTOB4+MTOB5+MTOB6',

                #----------
                #TIB+TOB
                #----------
                'MTIB4+TOB1+TOB2',
                'MTIB4+TOB2+MTOB3',
                'MTIB3+TOB1+TOB2',

                #----------
                #TID+TOB
                #----------

                'MTID1_pos+TOB1+TOB2','MTID1_neg+TOB1+TOB2',
                'MTID1_pos+TOB1+TOB2','MTID1_neg+TOB1+TOB2',
                'MTID2_pos+TOB1+TOB2','MTID2_neg+TOB1+TOB2',
                'MTID3_pos+TOB1+TOB2','MTID3_neg+TOB1+TOB2',

                #TOB+MTEC
                'TOB1+TOB2+MTEC1_pos','TOB1+TOB2+MTEC1_neg',

                #TID+TEC
                'TID1+TID2+TEC1_pos', 'TID1+TID2+TEC1_neg', 
                'TID2+MTID3+TEC1_pos', 'TID2+MTID3+TEC1_neg',
                'MTID3+TEC1_pos+MTEC2_pos', 'MTID3+TEC1_neg+MTEC2_neg'), 


        TOB = cms.PSet(
	 TTRHBuilder = cms.string('WithTrackAngle'),
	 clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
	 matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
	 skipClusters   = cms.InputTag('displacedGeneralStepClusters')
        ),

        MTOB = cms.PSet(
         TTRHBuilder = cms.string('WithTrackAngle'),
	 clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
         rphiRecHits    = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
         skipClusters   = cms.InputTag('displacedGeneralStepClusters')
        ),

        TIB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'),
	 clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
         matchedRecHits = cms.InputTag('siStripMatchedRecHits','matchedRecHit'),
         skipClusters   = cms.InputTag('displacedGeneralStepClusters')
        ),

        MTIB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'), 
	 clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
	 rphiRecHits    = cms.InputTag('siStripMatchedRecHits','rphiRecHit'),
         skipClusters   = cms.InputTag('displacedGeneralStepClusters')
        ),

        TID = cms.PSet(
	 TTRHBuilder = cms.string('WithTrackAngle'),
	 clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
	 matchedRecHits = cms.InputTag('siStripMatchedRecHits','matchedRecHit'),
	 skipClusters   = cms.InputTag('displacedGeneralStepClusters'),
         useRingSlector = cms.bool(True),
	 minRing = cms.int32(1),
         maxRing = cms.int32(2)
	),
        
	MTID = cms.PSet(
         TTRHBuilder = cms.string('WithTrackAngle'),
 	 clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
	 rphiRecHits    = cms.InputTag('siStripMatchedRecHits','rphiRecHit'),
	 skipClusters = cms.InputTag('displacedGeneralStepClusters'),
	 useRingSlector = cms.bool(True),
	 minRing = cms.int32(3),
         maxRing = cms.int32(3)
        ),

        TEC = cms.PSet(
	 TTRHBuilder = cms.string('WithTrackAngle'),
	 clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
	 matchedRecHits = cms.InputTag('siStripMatchedRecHits','matchedRecHit'),
	 skipClusters = cms.InputTag('displacedGeneralStepClusters'),
         useRingSlector = cms.bool(True),
	 minRing = cms.int32(5),
         maxRing = cms.int32(5)
	),

        MTEC = cms.PSet(
	 TTRHBuilder = cms.string('WithTrackAngle'),
	 clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTight')),
	 rphiRecHits = cms.InputTag('siStripMatchedRecHits','rphiRecHit'),
	 skipClusters = cms.InputTag('displacedGeneralStepClusters'),
	 useRingSlector = cms.bool(True),
	 minRing = cms.int32(6),
         maxRing = cms.int32(7)
	) 
) 
