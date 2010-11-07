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

regionalCosmicTrackerSeeds = cms.EDProducer( "SeedGeneratorFromRegionHitsEDProducer",
   RegionFactoryPSet = cms.PSet(                                 
      ComponentName = cms.string( "CosmicRegionalSeedGenerator" ),
      RegionPSet = cms.PSet(
        ptMin          = cms.double( 1.0 ),
        rVertex        = cms.double( 5 ),
        zVertex        = cms.double( 5 ),
        deltaEtaRegion = cms.double( 0.1 ),
        deltaPhiRegion = cms.double( 0.1 ),
        precise        = cms.bool( True ),
        measurementTrackerName = cms.string('')
        ),
      ToolsPSet = cms.PSet(
        thePropagatorName           = cms.string("AnalyticalPropagator"),
        regionBase                  = cms.string("seedOnCosmicMuon")
        #regionBase                  = cms.string("seedOnStaMuon")
        ),
      CollectionsPSet = cms.PSet(
        recoMuonsCollection            = cms.InputTag("muons"),  # se to "muons" and change ToolsPSet.regionBase to "" in order to use these.
        recoTrackMuonsCollection       = cms.InputTag("cosmicMuons")
        )
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string( "GenericPairGenerator"),
        #ComponentName = cms.string( "GenericTripletGenerator"),
        LayerPSet = cms.PSet(
           layerInfo,
           layerList = cms.vstring('TOB6+TOB5',
                                   'TOB6+TOB4', 
                                   'TOB6+TOB3',
                                   'TOB5+TOB4',
                                   'TOB5+TOB3',
                                   'TOB4+TOB3',
                                   'TEC1_neg+TOB6',
                                   'TEC1_neg+TOB5',
                                   'TEC1_neg+TOB4',
                                   'TEC1_pos+TOB6',
                                   'TEC1_pos+TOB5',
                                   'TEC1_pos+TOB4'                                   
                                   )
           ),
        ##PropagationDirection = cms.string('alongMomentum'),
        ##NavigationDirection = cms.string('outsideIn')
    ), 

    ClusterCheckPSet = cms.PSet (
      MaxNumberOfCosmicClusters = cms.uint32(3000),
      ClusterCollectionLabel = cms.InputTag( "siStripClusters" ),
      MaxNumberOfPixelClusters = cms.uint32(3000),
      PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
      doClusterCheck = cms.bool( True ),
      #this is dangerous, but what the heck
      silentClusterCheck = cms.untracked.bool(True)
    ) ,

    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),

    TTRHBuilder = cms.string( "WithTrackAngle" ) ,

    SeedCreatorPSet = cms.PSet(
      ComponentName = cms.string('CosmicSeedCreator'),
      propagator = cms.string('PropagatorWithMaterial'),
      maxseeds = cms.int32(10000)
      )

          
)

