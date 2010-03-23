import FWCore.ParameterSet.Config as cms

layerInfo = cms.PSet(
    TOB = cms.PSet(
      TTRHBuilder = cms.string('WithTrackAngle'),
      ),
    
    TEC = cms.PSet(
      #useSimpleRphiHitsCleaner = cms.untracked.bool(True),
      minRing = cms.int32(6),
      useRingSlector = cms.bool(False),
      TTRHBuilder = cms.string('WithTrackAngle'),
      maxRing = cms.int32(7)
      )
)

regionalCosmicMuonSeeds = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
   RegionFactoryPSet = cms.PSet(                                 
      ComponentName = cms.string( "CosmicRegionalSeedGenerator" ),
      RegionPSet = cms.PSet(
        tp_label       = cms.InputTag("mergedtruth","MergedTrackTruth"),
        ptMin          = cms.double( 1.0 ),
        rVertex        = cms.double( 5 ),
        zVertex        = cms.double( 5 ),
        deltaEtaRegion = cms.double( 0.1 ),
        deltaPhiRegion = cms.double( 0.1 ),
        precise        = cms.bool( True ),
        ),
      HLTPSet = cms.PSet(
        hltTag               = cms.InputTag("hltDiMuonL2PreFiltered0","",""),
        triggerSummaryLabel  = cms.string("hltTriggerSummaryAOD"),
        thePropagatorName    = cms.string("AnalyticalPropagator"),
        seeding              = cms.string("")
        ),
      CollectionsForSeedingPSet = cms.PSet(
        l2MuonsRecoTracksCollection = cms.InputTag("hltL2Muons"),
        staMuonsCollection          = cms.InputTag("muons")
      )
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string( "GenericPairGenerator"),
        #ComponentName = cms.string( "GenericTripletGenerator"),
        LayerPSet = cms.PSet(
           layerInfo,
           layerList = cms.vstring(#'TOB6+TOB5+TOB4',
                                   'TOB6+TOB5',
                                   'TOB6+TOB4', 
                                   'TOB6+TOB3',
                                   'TOB5+TOB4',
                                   'TOB5+TOB3',
                                   'TOB4+TOB3',
                                   )
           ),
        ##PropagationDirection = cms.string('alongMomentum'),
        ##NavigationDirection = cms.string('outsideIn')
    ), 

    ClusterCheckPSet = cms.PSet (
      MaxNumberOfCosmicClusters = cms.double( 50000 ),
      ClusterCollectionLabel = cms.InputTag( "hltSiStripClusters" ),
      doClusterCheck = cms.bool( False )
    ) ,

    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),

    TTRHBuilder = cms.string( "WithTrackAngle" ) ,

    SeedCreatorPSet = cms.PSet(
      ComponentName = cms.string('CosmicSeedCreator'),
      propagator = cms.string('PropagatorWithMaterial'),
    )                                 
)

