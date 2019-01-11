import FWCore.ParameterSet.Config as cms

HGVHistoProducerAlgoBlock = cms.PSet(

    minEta = cms.double(-4.5),
    maxEta = cms.double(4.5),
    nintEta = cms.int32(100),
    useFabsEta = cms.bool(False),

    #parameters for calo particles energy 
    minCaloEne  = cms.double(0.),
    maxCaloEne  = cms.double(500.),
    nintCaloEne = cms.int32(250),

    #parameters for calo particles pt 
    minCaloPt  = cms.double(0.),
    maxCaloPt  = cms.double(100.),
    nintCaloPt = cms.int32(100),

    #parameters for calo particles phi
    minCaloPhi  = cms.double(-4.),
    maxCaloPhi  = cms.double(4.),
    nintCaloPhi = cms.int32(100),

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

    #Parameters for the energy density of cluster cells per thickness 
    minCellsEneDensperthick = cms.double(0.),
    maxCellsEneDensperthick = cms.double(100.),
    nintCellsEneDensperthick = cms.int32(100)





)

