# The following comments couldn't be translated into the new config version:

#along momentum

#and opposite to momentum

import FWCore.ParameterSet.Config as cms

layerInfo = cms.PSet(
    TID = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        HitProducer = cms.string('siPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    ),
    TEC = cms.PSet(
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle'),
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        maxRing = cms.int32(7)
    )
)
combinatorialbeamhaloseedfinder = cms.EDFilter("CtfSpecialSeedGenerator",
    ErrorRescaling = cms.double(50.0),
    OrderedHitsFactoryPSets = cms.VPSet(cms.PSet(
        ComponentName = cms.string('BeamHaloPairGenerator'),
        maxTheta = cms.double(1.0),
        PropagationDirection = cms.string('alongMomentum'),
        NavigationDirection = cms.string('outsideIn'),
        LayerPSet = cms.PSet(
            layerInfo,
            layerList = cms.vstring('FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg', 'TID2_pos+TID3_pos', 'TID2_neg+TID3_neg', 'TEC7_pos+TEC8_pos', 'TEC8_pos+TEC9_pos', 'TEC7_neg+TEC8_neg', 'TEC8_neg+TEC9_neg')
        )
    ), cms.PSet(
        ComponentName = cms.string('BeamHaloPairGenerator'),
        maxTheta = cms.double(1.0),
        PropagationDirection = cms.string('oppositeToMomentum'),
        NavigationDirection = cms.string('outsideIn'),
        LayerPSet = cms.PSet(
            layerInfo,
            layerList = cms.vstring('FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg', 'TID2_pos+TID3_pos', 'TID2_neg+TID3_neg', 'TEC7_pos+TEC8_pos', 'TEC8_pos+TEC9_pos', 'TEC7_neg+TEC8_neg', 'TEC8_neg+TEC9_neg')
        )
    )),
    Charges = cms.vint32(-1, 1),
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
    UseScintillatorsConstraint = cms.bool(False),
    TTRHBuilder = cms.string('WithTrackAngle'),
    SeedsFromPositiveY = cms.bool(False),
    CheckHitsAreOnDifferentLayers = cms.bool(False),
    SeedMomentum = cms.double(15.0), ##initial momentum in GeV !!!set to a lower value for slice test data

    SetMomentum = cms.bool(True)
)


