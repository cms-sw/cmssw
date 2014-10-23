import FWCore.ParameterSet.Config as cms

OuterTrackerClusters = cms.EDAnalyzer('OuterTrackerClusters',
    
    TopFolderName = cms.string('OuterTrackerV'),
    
# Cluster Stacks
    TH1TTCluster_Stack = cms.PSet(
        Nbinsx = cms.int32(6),
        xmax = cms.double(6.5),                      
        xmin = cms.double(0.5)
        ),
          
)
