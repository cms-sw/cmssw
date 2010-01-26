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
        tp_label       = cms.InputTag("mergedtruth","MergedTrackTruth","GenSimRaw"),
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
        )
      ),
                                          
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string( "GenericPairGenerator"),
        #ComponentName = cms.string( "GenericTripletGenerator"),
        LayerPSet = cms.PSet(
           layerInfo,
           layerList = cms.vstring(#'TOB6+TOB5+TOB4',
                                   'TOB5+TOB6',
                                   'TOB4+TOB6', 
                                   'TOB4+TOB5'
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

def reverseHit(layerList):
    newLayerList= cms.vstring()
    import string
    for setup in layerList:
        l=setup.split("+")
        l.reverse()
        newLayerList.append(string.join(l,"+"))
    return newLayerList

def addTECLayers(module):
    import FWCore.ParameterSet.Config as cms
    module.OrderedHitsFactoryPSet.LayerPSet.TEC = cms.PSet(
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(6),
        maxRing = cms.int32(7)
        )
    
    module.OrderedHitsFactoryPSet.LayerPSet.TEC8 = cms.PSet(
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle')
        )
    module.OrderedHitsFactoryPSet.LayerPSet.TEC9 = cms.PSet(
        useRingSlector = cms.bool(False),
        TTRHBuilder = cms.string('WithTrackAngle')
        )

    for side in ['_pos','_neg']:
        module.OrderedHitsFactoryPSet.LayerPSet.layerList.extend([
            "TOB5+TEC1"+side,
            "TOB6+TEC1"+side,
            "TEC8"+side+"+TEC9"+side,
            "TEC7"+side+"+TEC8"+side,
            "TEC1"+side+"+TEC2"+side,
            "TEC2"+side+"+TEC3"+side,
            "TEC3"+side+"+TEC4"+side,
            "TEC4"+side+"+TEC5"+side,
            "TEC5"+side+"+TEC6"+side,
            "TEC6"+side+"+TEC7"+side,
            "TEC7"+side+"+TEC8"+side,
            "TEC8"+side+"+TEC9"+side
            ])
#from cosmicLayers import addTECLayers
addTECLayers(regionalCosmicMuonSeeds)
regionalCosmicMuonSeeds.OrderedHitsFactoryPSet.LayerPSet.layerList = reverseHit(regionalCosmicMuonSeeds.OrderedHitsFactoryPSet.LayerPSet.layerList)

