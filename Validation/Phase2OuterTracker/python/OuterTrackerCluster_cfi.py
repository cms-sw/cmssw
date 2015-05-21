import FWCore.ParameterSet.Config as cms

OuterTrackerCluster = cms.EDAnalyzer('OuterTrackerCluster',
    
    TopFolderName = cms.string('Phase2OuterTrackerV'),
    TTClusters       = cms.InputTag("TTClustersFromPixelDigis", "ClusterInclusive"),
    TTClusterMCTruth = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterInclusive"),    
    verbosePlots   = cms.untracked.bool(False),


# Cluster Barrel Layers    
    TH1TTCluster_Layers = cms.PSet(
        Nbinsx = cms.int32(6),
        xmax = cms.double(6.5),                      
        xmin = cms.double(0.5)
        ),
    
# Cluster EC Disks    
    TH1TTCluster_Disks = cms.PSet(
        Nbinsx = cms.int32(5),
        xmax = cms.double(5.5),                      
        xmin = cms.double(0.5)
        ),
    
# Cluster EC Rings
    TH1TTCluster_Rings = cms.PSet(
        Nbinsx = cms.int32(16),
        xmax = cms.double(16.5),                      
        xmin = cms.double(0.5)
        ),

# Cluster Eta
    TH1TTCluster_Eta = cms.PSet(
        Nbinsx = cms.int32(45),
        xmax = cms.double(3),                      
        xmin = cms.double(-3)
        ),          

)
