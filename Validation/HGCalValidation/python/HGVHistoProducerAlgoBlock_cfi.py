import FWCore.ParameterSet.Config as cms

HGVHistoProducerAlgoBlock = cms.PSet(

    minEta = cms.double(-4.5),
    maxEta = cms.double(4.5),
    nintEta = cms.int32(100),
    useFabsEta = cms.bool(False),

    #z position of vertex 
    minZpos = cms.double(-550.),
    maxZpos = cms.double(550.),
    nintZpos = cms.int32(1100),

    #Parameters for the total number of layer clusters per layer
    minTotNClsperlay = cms.double(0.),
    maxTotNClsperlay = cms.double(50.),
    nintTotNClsperlay = cms.int32(50),
                             
    #Parameters for the total number of layer clusters per thickness
    minTotNClsperthick = cms.double(0.),
    maxTotNClsperthick = cms.double(800.),
    nintTotNClsperthick = cms.int32(100),

    #Parameters for the total number of cells per per thickness per layer
    minTotNcellsperthickperlayer = cms.double(0.),
    maxTotNcellsperthickperlayer = cms.double(10000.),
    nintTotNcellsperthickperlayer = cms.int32(100),

    #Parameters for the distance of cluster cells to seed cell per thickness per layer
    minDisToSeedperthickperlayer  = cms.double(0.),
    maxDisToSeedperthickperlayer  = cms.double(1000.),
    nintDisToSeedperthickperlayer = cms.int32(100),

    #Parameters for the energy weighted distance of cluster cells to seed cell per thickness per layer
    minDisToSeedperthickperlayerenewei  = cms.double(0.),
    maxDisToSeedperthickperlayerenewei  = cms.double(1.),
    nintDisToSeedperthickperlayerenewei = cms.int32(10),

    #Parameters for the distance of cluster cells to max cell per thickness per layer
    minDisToMaxperthickperlayer  = cms.double(0.),
    maxDisToMaxperthickperlayer  = cms.double(10.),
    nintDisToMaxperthickperlayer = cms.int32(10),

    #Parameters for the energy weighted distance of cluster cells to max cell per thickness per layer
    minDisToMaxperthickperlayerenewei  = cms.double(0.),
    maxDisToMaxperthickperlayerenewei  = cms.double(1.),
    nintDisToMaxperthickperlayerenewei = cms.int32(10),

    #Parameters for the energy density of cluster cells per thickness 
    minCellsEneDensperthick = cms.double(0.),
    maxCellsEneDensperthick = cms.double(100.),
    nintCellsEneDensperthick = cms.int32(100)





)

