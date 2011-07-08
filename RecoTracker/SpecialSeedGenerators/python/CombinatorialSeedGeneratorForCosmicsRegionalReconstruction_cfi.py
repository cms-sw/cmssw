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
        useSimpleRphiHitsCleaner = cms.bool(True),
        minRing = cms.int32(5),
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        useRingSlector = cms.bool(False),
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

def makeSimpleCosmicSeedLayers(*layers):
    layerList = cms.vstring()
    if 'ALL' in layers: 
        layers = [ 'TOB', 'TEC', 'TOBTEC', 'TECSKIP' ]
    if 'TOB' in layers:
        layerList += ['TOB4+TOB5+TOB6',
                      'TOB3+TOB5+TOB6',
                      'TOB3+TOB4+TOB5',
                      'TOB3+TOB4+TOB6',
                      'TOB2+TOB4+TOB5',
                      'TOB2+TOB3+TOB5']
    if 'TEC' in layers:
        TECwheelTriplets = [ (i,i+1,i+2) for i in range(7,0,-1)]
        layerList += [ 'TEC%d_pos+TEC%d_pos+TEC%d_pos' % ls for ls in TECwheelTriplets ]
        layerList += [ 'TEC%d_neg+TEC%d_neg+TEC%d_neg' % ls for ls in TECwheelTriplets ]
    if 'TECSKIP' in layers:
        TECwheelTriplets = [ (i-1,i+1,i+2) for i in range(7,1,-1)] + [ (i-1,i,i+2) for i in range(7,1,-1)  ]
        layerList += [ 'TEC%d_pos+TEC%d_pos+TEC%d_pos' % ls for ls in TECwheelTriplets ]
        layerList += [ 'TEC%d_neg+TEC%d_neg+TEC%d_neg' % ls for ls in TECwheelTriplets ]
    if 'TOBTEC' in layers:
        layerList += [ 'TOB6+TEC1_pos+TEC2_pos',
                       'TOB6+TEC1_neg+TEC2_neg',
                       'TOB6+TOB5+TEC1_pos',
                       'TOB6+TOB5+TEC1_neg' ]
    #print "SEEDING LAYER LIST = ", layerList
    return layerList


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
        recoMuonsCollection            = cms.InputTag(""),  # se to "muons" and change ToolsPSet.regionBase to "" in order to use these
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
        ComponentName = cms.string('GenericTripletGenerator'),
        LayerPSet = cms.PSet(
            layerInfo,
            layerList = makeSimpleCosmicSeedLayers('ALL'),
        ),
        PropagationDirection = cms.string('alongMomentum'),
        NavigationDirection = cms.string('outsideIn')
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

