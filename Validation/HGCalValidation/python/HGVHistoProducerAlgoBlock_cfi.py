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
    minScore = cms.double(0.),
    maxScore = cms.double(4.01),
    nintScore = cms.int32(200),

    #Parameters for shared energy fraction. That is:
    #1. Fraction of each of the layer clusters energy related to a
    #   calo particle over that calo particle's energy.
    #2. Fraction of each of the calo particles energy
    #   related to a layer cluster over that layer cluster's energy.
    minSharedEneFrac = cms.double(0.),
    maxSharedEneFrac = cms.double(1.),
    nintSharedEneFrac = cms.int32(100),

    #Same as above for multiclusters
    minMCLSharedEneFrac = cms.double(0.),
    maxMCLSharedEneFrac = cms.double(1.0),
    nintMCLSharedEneFrac = cms.int32(100),

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
    nintCellsEneDensperthick = cms.int32(200),

    #Parameters for the total number of multiclusters per event
    #We always treet one event as two events, one in +z one in -z
    minTotNMCLs = cms.double(0.),
    maxTotNMCLs = cms.double(50.),
    nintTotNMCLs = cms.int32(50),

    #Parameters for the total number of layer clusters in multicluster
    minTotNClsinMCLs = cms.double(0.),
    maxTotNClsinMCLs = cms.double(400.),
    nintTotNClsinMCLs = cms.int32(100),

    #Parameters for the total number of layer clusters in multicluster per layer
    minTotNClsinMCLsperlayer = cms.double(0.),
    maxTotNClsinMCLsperlayer = cms.double(50.),
    nintTotNClsinMCLsperlayer = cms.int32(50),

    #Parameters for the multiplicity of layer clusters in multicluster
    minMplofLCs = cms.double(0.),
    maxMplofLCs = cms.double(20.),
    nintMplofLCs = cms.int32(20),

    #Parameters for cluster size
    minSizeCLsinMCLs = cms.double(0.),
    maxSizeCLsinMCLs = cms.double(50.),
    nintSizeCLsinMCLs = cms.int32(50),

    #Parameters for the energy of a cluster per multiplicity
    minClEnepermultiplicity  = cms.double(0.),
    maxClEnepermultiplicity = cms.double(10.),
    nintClEnepermultiplicity = cms.int32(10),

    #parameters for X
    minX  = cms.double(-300.),
    maxX  = cms.double(300.),
    nintX = cms.int32(100),

    #parameters for Y
    minY  = cms.double(-300.),
    maxY  = cms.double(300.),
    nintY = cms.int32(100),

    #parameters for Z
    minZ  = cms.double(-550.),
    maxZ  = cms.double(550.),
    nintZ = cms.int32(1100)

)

