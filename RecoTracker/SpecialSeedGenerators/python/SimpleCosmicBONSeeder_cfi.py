import FWCore.ParameterSet.Config as cms

from RecoTracker.SpecialSeedGenerators.CombinatorialSeedGeneratorForCosmics_cfi import layerInfo

def makeSimpleCosmicSeedLayers(*layers):
    layerList = cms.vstring()
    if 'ALL' in layers: 
        layers = [ 'TOB', 'TEC', 'TOBTEC', 'TECSKIP' ]
    if 'TOB' in layers:
        layerList += ['TOB4+TOB5+TOB6',
                      'TOB3+TOB5+TOB6',
                      'TOB3+TOB4+TOB5',
                      'TOB3+TOB4+TOB6']
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
    

simpleCosmicBONSeeds = cms.EDProducer("SimpleCosmicBONSeeder",
    TTRHBuilder = cms.string('WithTrackAngle'),
    ClusterCheckPSet = cms.PSet(
            doClusterCheck = cms.bool(True),
            MaxNumberOfCosmicClusters = cms.uint32(300),
            ClusterCollectionLabel = cms.InputTag("siStripClusters"),
    ),
    RegionPSet = cms.PSet(
        originZPosition  = cms.double(0.0),
        originRadius     = cms.double(150.0),
        originHalfLength = cms.double(90.0),
        ptMin = cms.double(0.9),
    ),
    TripletsPSet = cms.PSet(
        layerInfo,
        layerList = makeSimpleCosmicSeedLayers('ALL'),
        debugLevel = cms.untracked.uint32(0),  # debug triplet finding (0 to 3)
    ),
    #rescaleError    = cms.double(1),   # rescale seed error (a factor 50 was used historically for cosmics)
    rescaleError    = cms.double(50),   # this rescaling has to be avoided. TO BE FIXED

                                      
    writeTriplets   = cms.bool(False), # write the triplets to the Event as OwnVector<TrackingRecHit>
    helixDebugLevel = cms.untracked.uint32(0), # debug FastHelix (0 to 2)
    seedDebugLevel  = cms.untracked.uint32(0), # debug seed building (0 to 3)
)

