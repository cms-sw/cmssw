import FWCore.ParameterSet.Config as cms

OuterTrackerClusters = cms.EDAnalyzer('OuterTrackerClusters',
                                     
    ClusterProducerStrip = cms.InputTag('siStripClusters'),
    
    TopFolderName = cms.string('OuterTrackerV'),
    
# Cluster Stacks
    TH1TTCluster_Stack = cms.PSet(
        Nbinsx = cms.int32(12),
        xmax = cms.double(11.5),                      
        xmin = cms.double(-0.5)
        ),
          
)
