import FWCore.ParameterSet.Config as cms

layerInfo = cms.PSet(
    TOB = cms.PSet(
      TTRHBuilder = cms.string('WithTrackAngle'),
      ),
    TEC = cms.PSet(
      minRing = cms.int32(6),
      useRingSlector = cms.bool(False),
      TTRHBuilder = cms.string('WithTrackAngle'),
      maxRing = cms.int32(7)
      )
)
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
        regionBase                  = cms.string("seedOnCosmicMuon") # seedOnL2Muon or seedOnCosmicMuon or seedOnStaMuon(default)

        ),
      CollectionsPSet = cms.PSet(
        recoMuonsCollection            = cms.InputTag(""),  # se to "muons" and change ToolsPSet.regionBase to "" in order to use these.
        recoTrackMuonsCollection       = cms.InputTag("cosmicMuons"), # or cosmicMuons1Leg and change ToolsPSet.regionBase to "seedOnCosmicMuon" in order to use these.
        recoL2MuonsCollection          = cms.InputTag(""), # given by the hlt path sequence
        ),
      RegionInJetsCheckPSet = cms.PSet( # verify if the region is built inside a jet
        doJetsExclusionCheck   = cms.bool( True ),
        deltaRExclusionSize    = cms.double( 0.3 ),
        jetsPtMin              = cms.double( 5 ),
        recoCaloJetsCollection = cms.InputTag("ak5CaloJets")
        )
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string( "GenericPairGenerator"),
        LayerSrc = cms.InputTag("regionalCosmicTrackerSeedingLayers")
    ), 

    ClusterCheckPSet = cms.PSet (
      MaxNumberOfCosmicClusters = cms.uint32(10000),
      ClusterCollectionLabel = cms.InputTag( "siStripClusters" ),
      MaxNumberOfPixelClusters = cms.uint32(10000),
      PixelClusterCollectionLabel = cms.InputTag("siPixelClusters"),
      doClusterCheck = cms.bool( False )
    ) ,

    SeedComparitorPSet = cms.PSet(  ComponentName = cms.string( "none" ) ),

    TTRHBuilder = cms.string( "WithTrackAngle" ) ,

    SeedCreatorPSet = cms.PSet(
      ComponentName = cms.string('CosmicSeedCreator'),
      propagator = cms.string('PropagatorWithMaterial'),
      maxseeds = cms.int32(10000)
      )

          
)

