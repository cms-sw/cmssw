import FWCore.ParameterSet.Config as cms

HGVHistoProducerAlgoBlock = cms.PSet(

    minEta = cms.double(-4.5),
    maxEta = cms.double(4.5),
    nintEta = cms.int32(100),
    useFabsEta = cms.bool(False),

    #parameters for energy 
    minEne  = cms.double(0.),
    maxEne  = cms.double(500.),
    nintEne = cms.int32(250),

    #parameters for pt 
    minPt  = cms.double(0.),
    maxPt  = cms.double(100.),
    nintPt = cms.int32(100),

    #parameters for phi
    minPhi  = cms.double(-4.),
    maxPhi  = cms.double(4.),
    nintPhi = cms.int32(100),

    #parameters for counting mixed hits clusters
    minMixedHitsCluster = cms.double(0.),
    maxMixedHitsCluster = cms.double(800.),
    nintMixedHitsCluster = cms.int32(100),

    #parameters for the total amount of energy clustered by all layer clusters (fraction over caloparticles)
    minEneCl = cms.double(0.),
    maxEneCl = cms.double(110.),
    nintEneCl = cms.int32(110),

    #parameters for the longitudinal depth barycenter 
    minLongDepBary = cms.double(0.),
    maxLongDepBary = cms.double(110.),
    nintLongDepBary = cms.int32(110),

    #z position of vertex 
    minZpos = cms.double(-550.),
    maxZpos = cms.double(550.),
    nintZpos = cms.int32(1100),

    #Parameters for the total number of layer clusters per layer
    minTotNClsperlay = cms.double(0.),
    maxTotNClsperlay = cms.double(50.),
    nintTotNClsperlay = cms.int32(50),
                             
    #Parameters for the energy clustered by layer clusters per layer (fraction)
    minEneClperlay = cms.double(0.),
    maxEneClperlay = cms.double(110.),
    nintEneClperlay = cms.int32(110),

    #Parameters for the score both for: 
    #1. calo particle to layer clusters association per layer
    #2. layer cluster to calo particles association per layer
    minScore = cms.double(-1.01),
    maxScore = cms.double(1.01),
    nintScore = cms.int32(200),
             
    #Parameters for shared energy fraction. That is: 
    #1. Fraction of each of the layer clusters energy related to a 
    #   calo particle over that calo particle's energy.
    #2. Fraction of each of the calo particles energy 
    #   related to a layer cluster over that layer cluster's energy.
    minSharedEneFrac = cms.double(0.),
    maxSharedEneFrac = cms.double(1.),
    nintSharedEneFrac = cms.int32(100),

    #Parameters for the total number of layer clusters per thickness
    minTotNClsperthick = cms.double(0.),
    maxTotNClsperthick = cms.double(800.),
    nintTotNClsperthick = cms.int32(100),

    #Parameters for the total number of cells per per thickness per layer
    minTotNcellsperthickperlayer = cms.double(0.),
    maxTotNcellsperthickperlayer = cms.double(500.),
    nintTotNcellsperthickperlayer = cms.int32(100),

    #Parameters for the distance of cluster cells to seed cell per thickness per layer
    minDisToSeedperthickperlayer  = cms.double(0.),
    maxDisToSeedperthickperlayer  = cms.double(300.),
    nintDisToSeedperthickperlayer = cms.int32(100),

    #Parameters for the energy weighted distance of cluster cells to seed cell per thickness per layer
    minDisToSeedperthickperlayerenewei  = cms.double(0.),
    maxDisToSeedperthickperlayerenewei  = cms.double(10.),
    nintDisToSeedperthickperlayerenewei = cms.int32(50),

    #Parameters for the distance of cluster cells to max cell per thickness per layer
    minDisToMaxperthickperlayer  = cms.double(0.),
    maxDisToMaxperthickperlayer  = cms.double(300.),
    nintDisToMaxperthickperlayer = cms.int32(100),

    #Parameters for the energy weighted distance of cluster cells to max cell per thickness per layer
    minDisToMaxperthickperlayerenewei  = cms.double(0.),
    maxDisToMaxperthickperlayerenewei  = cms.double(50.),
    nintDisToMaxperthickperlayerenewei = cms.int32(50),

    #Parameters for the distance of cluster cells to max cell per thickness per layer
    minDisSeedToMaxperthickperlayer  = cms.double(0.),
    maxDisSeedToMaxperthickperlayer  = cms.double(300.),
    nintDisSeedToMaxperthickperlayer = cms.int32(100),

    #Parameters for the energy of a cluster per thickness per layer
    minClEneperthickperlayer  = cms.double(0.),
    maxClEneperthickperlayer = cms.double(10.),
    nintClEneperthickperlayer = cms.int32(100),

    #Parameters for the energy density of cluster cells per thickness 
    minCellsEneDensperthick = cms.double(0.),
    maxCellsEneDensperthick = cms.double(100.),
    nintCellsEneDensperthick = cms.int32(200)

)

