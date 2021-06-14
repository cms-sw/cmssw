# The following comments couldn't be translated into the new config version:

#"StandardHitPairGenerator"
import FWCore.ParameterSet.Config as cms

from RecoTracker.TkTrackingRegions.GlobalTrackingRegion_cfi import *
from RecoLocalTracker.SiStripClusterizer.SiStripClusterChargeCut_cfi import *

layerInfo = cms.PSet(
    MTIB = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone')),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TIB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone'))
    ),
    MTOB = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone')),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle'),
        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone'))
    ),
    TEC = cms.PSet(
        useSimpleRphiHitsCleaner = cms.bool(True),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone')),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(7)
    )
)
combinatorialcosmicseedingtripletsTOB_layerList = cms.vstring('MTOB4+MTOB5+MTOB6',
    'MTOB3+MTOB5+MTOB6',
    'MTOB3+MTOB4+MTOB5',
    'TOB2+MTOB4+MTOB5',
    'MTOB3+MTOB4+MTOB6',
    'TOB2+MTOB4+MTOB6')
combinatorialcosmicseedingpairsTECpos_layerList = cms.vstring('TEC1_pos+TEC2_pos',
    'TEC2_pos+TEC3_pos',
    'TEC3_pos+TEC4_pos',
    'TEC4_pos+TEC5_pos',
    'TEC5_pos+TEC6_pos',
    'TEC6_pos+TEC7_pos',
    'TEC7_pos+TEC8_pos',
    'TEC8_pos+TEC9_pos')
combinatorialcosmicseedingtripletsTIB_layerList = cms.vstring('TIB1+TIB2+MTIB3')

combinatorialcosmicseedfinder = cms.EDProducer("CtfSpecialSeedGenerator",
    SeedMomentum = cms.double(5.0), ##initial momentum in GeV !!!set to a lower value for slice test data
    ErrorRescaling = cms.double(50.0),
    RegionFactoryPSet = cms.PSet(
        RegionPSetBlock,
        ComponentName = cms.string('GlobalRegionProducer')
    ),
    UpperScintillatorParameters = cms.PSet(
        LenghtInZ = cms.double(100.0),
        GlobalX = cms.double(0.0),
        GlobalZ = cms.double(50.0),
        WidthInX = cms.double(100.0),
        GlobalY = cms.double(300.0)
    ),
    Charges = cms.vint32(-1),
    OrderedHitsFactoryPSets = cms.VPSet(
        cms.PSet(
            ComponentName = cms.string('GenericTripletGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingtripletsTOB"),
            PropagationDirection = cms.string('alongMomentum'),
            NavigationDirection = cms.string('outsideIn')
        ), 
        cms.PSet(
            ComponentName = cms.string('GenericPairGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingpairsTECpos"),
            PropagationDirection = cms.string('alongMomentum'),
            NavigationDirection = cms.string('outsideIn')
        ), 
        cms.PSet(
            ComponentName = cms.string('GenericTripletGenerator'),
            LayerSrc = cms.InputTag("combinatorialcosmicseedingtripletsTIB"),
            PropagationDirection = cms.string('oppositeToMomentum'),
            NavigationDirection = cms.string('insideOut')
        )
    ),
    UseScintillatorsConstraint = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    LowerScintillatorParameters = cms.PSet(
        LenghtInZ = cms.double(100.0),
        GlobalX = cms.double(0.0),
        GlobalZ = cms.double(50.0),
        WidthInX = cms.double(100.0),
        GlobalY = cms.double(-100.0)
    ),
    SeedsFromPositiveY = cms.bool(True),
    #***top-bottom                                         
    SeedsFromNegativeY = cms.bool(False),
    #***
    doClusterCheck = cms.bool(True),
    DontCountDetsAboveNClusters = cms.uint32(20),
    CheckHitsAreOnDifferentLayers = cms.bool(False),
    SetMomentum = cms.bool(True),
    ClusterCollectionLabel = cms.InputTag("siStripClusters"),
    MaxNumberOfCosmicClusters = cms.uint32(300),
    PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
    MaxNumberOfPixelClusters = cms.uint32(300),
    requireBOFF = cms.bool(False),
    maxSeeds = cms.int32(10000),
)


