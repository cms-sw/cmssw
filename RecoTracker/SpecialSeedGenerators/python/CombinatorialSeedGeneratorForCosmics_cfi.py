# The following comments couldn't be translated into the new config version:

#"StandardHitPairGenerator"
#,
#"TIB1+TIB3"}

import FWCore.ParameterSet.Config as cms

layerInfo = cms.PSet(
    TIB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TIB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TIB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB6 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB1 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TOB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB2 = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    TEC = cms.PSet(
        useSimpleRphiHitsCleaner = cms.untracked.bool(True),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(7)
    ),
    TOB4 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TOB5 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    )
)
combinatorialcosmicseedfinder = cms.EDFilter("CtfSpecialSeedGenerator",
    ErrorRescaling = cms.double(50.0),
    OrderedHitsFactoryPSets = cms.VPSet(cms.PSet(
        ComponentName = cms.string('GenericTripletGenerator'),
        LayerPSet = cms.PSet(
            layerInfo,
            layerList = cms.vstring('TOB4+TOB5+TOB6', 'TOB3+TOB5+TOB6', 'TOB3+TOB4+TOB5', 'TOB2+TOB4+TOB5', 'TOB3+TOB4+TOB6', 'TOB2+TOB4+TOB6')
        ),
        PropagationDirection = cms.string('alongMomentum'),
        NavigationDirection = cms.string('outsideIn')
    ), cms.PSet(
        ComponentName = cms.string('GenericPairGenerator'),
        LayerPSet = cms.PSet(
            layerInfo,
            layerList = cms.vstring('TEC1_pos+TEC2_pos', 'TEC2_pos+TEC3_pos', 'TEC3_pos+TEC4_pos', 'TEC4_pos+TEC5_pos', 'TEC5_pos+TEC6_pos', 'TEC6_pos+TEC7_pos', 'TEC7_pos+TEC8_pos', 'TEC8_pos+TEC9_pos')
        ),
        PropagationDirection = cms.string('alongMomentum'),
        NavigationDirection = cms.string('outsideIn')
    ), cms.PSet(
        ComponentName = cms.string('GenericTripletGenerator'),
        LayerPSet = cms.PSet(
            layerInfo,
            layerList = cms.vstring('TIB1+TIB2+TIB3')
        ),
        PropagationDirection = cms.string('oppositeToMomentum'),
        NavigationDirection = cms.string('insideOut')
    )),
    UpperScintillatorParameters = cms.PSet(
        LenghtInZ = cms.double(100.0),
        GlobalX = cms.double(0.0),
        GlobalZ = cms.double(50.0),
        WidthInX = cms.double(100.0),
        GlobalY = cms.double(300.0)
    ),
    Charges = cms.vint32(-1),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(15.9),
            originRadius = cms.double(0.2),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.9),
            originXPos = cms.double(0.0),
            originZPos = cms.double(0.0)
        )
    ),
    UseScintillatorsConstraint = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    SeedsFromPositiveY = cms.bool(True),
    CheckHitsAreOnDifferentLayers = cms.bool(False),
    SeedMomentum = cms.double(1.0), ##initial momentum in GeV !!!set to a lower value for slice test data

    SetMomentum = cms.bool(True),
    LowerScintillatorParameters = cms.PSet(
        LenghtInZ = cms.double(100.0),
        GlobalX = cms.double(0.0),
        GlobalZ = cms.double(50.0),
        WidthInX = cms.double(100.0),
        GlobalY = cms.double(-100.0)
    )
)


