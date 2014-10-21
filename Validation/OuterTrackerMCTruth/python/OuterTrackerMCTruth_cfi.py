import FWCore.ParameterSet.Config as cms

OuterTrackerMCTruth = cms.EDAnalyzer('OuterTrackerMCTruth',
                                     
    ClusterProducerStrip = cms.InputTag('siStripClusters'),
    
    TopFolderName = cms.string('OuterTrackerV'),
    

# TPart Pt
    TH1TPart_Pt = cms.PSet(
        Nbinsx = cms.int32(50),
        xmax = cms.double(200.0),                      
        xmin = cms.double(0.0)
        ),

# TPart Eta/Phi
    TH1TPart_Angle_Pt10 = cms.PSet(
        Nbinsx = cms.int32(45),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(-3.1416)
        ),

# Sim Vertex XY
    TH1SimVtx_XY = cms.PSet(
        Nbinsx = cms.int32(30),
        xmax = cms.double(0.01),                      
        xmin = cms.double(-0.01),
        Nbinsy = cms.int32(30),
        ymax = cms.double(0.01),                      
        ymin = cms.double(-0.01)
        ),

# Sim Vertex RZ
    TH1SimVtx_RZ = cms.PSet(
        Nbinsx = cms.int32(30),
        xmax = cms.double(20.0),                      
        xmin = cms.double(-20.0),
        Nbinsy = cms.int32(30),
        ymax = cms.double(0.01),                      
        ymin = cms.double(0.0)
        ),

# CW vs. TPart AbsEta
    TH1TPart_AbsEta_CW = cms.PSet(
        Nbinsx = cms.int32(30),
        xmax = cms.double(3.0),                      
        xmin = cms.double(0.0)
        ),

# CW vs. TPart Eta
    TH1TPart_Eta_CW = cms.PSet(
        Nbinsx = cms.int32(50),
        xmax = cms.double(3.0),                      
        xmin = cms.double(-3.0)
        ),

# Stub modules vs. TPart AbsEta
    TH1TPart_AbsEta_PS2S = cms.PSet(
        Nbinsx = cms.int32(30),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(0.0)
        ),

# Stub modules vs. TPart Eta
    TH1TPart_Eta_PS2S = cms.PSet(
        Nbinsx = cms.int32(60),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(-3.1416)
        ),

## Cluster Stacks
    #TH1TTCluster_Stack = cms.PSet(
        #Nbinsx = cms.int32(12),
        #xmax = cms.double(11.5),                      
        #xmin = cms.double(-0.5)
        #),
          
)
