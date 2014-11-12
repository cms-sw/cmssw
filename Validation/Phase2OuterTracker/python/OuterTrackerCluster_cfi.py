import FWCore.ParameterSet.Config as cms

OuterTrackerCluster = cms.EDAnalyzer('OuterTrackerCluster',
    
    TopFolderName = cms.string('Phase2OuterTrackerV'),
    
# Cluster Stacks
    TH1TTCluster_Stack = cms.PSet(
        Nbinsx = cms.int32(6),
        xmax = cms.double(6.5),                      
        xmin = cms.double(0.5)
        ),

# Cluster Eta
    TH1TTCluster_Eta = cms.PSet(
        Nbinsx = cms.int32(45),
        xmax = cms.double(3),                      
        xmin = cms.double(-3)
        ),          

)
