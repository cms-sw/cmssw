import FWCore.ParameterSet.Config as cms

OuterTrackerMCTruth = cms.EDAnalyzer('OuterTrackerMCTruth',
    
    TopFolderName    = cms.string('Phase2OuterTrackerV'),
    TTClusters       = cms.InputTag("TTClustersFromPixelDigis", "ClusterInclusive"),
    TTClusterMCTruth = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterInclusive"),
    TTStubs          = cms.InputTag("TTStubsFromPixelDigis", "StubAccepted"),
    TTStubMCTruth    = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"),
    TTTracks         = cms.InputTag("TTTracksFromPixelDigis", "Level1TTTracks"),
    TTTrackMCTruth   = cms.InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks"),    
    HQDelim          = cms.int32(4),
    verbosePlots     = cms.untracked.bool(False),


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
    TH2SimVtx_XY = cms.PSet(
        Nbinsx = cms.int32(24),
        xmax = cms.double(0.006),                      
        xmin = cms.double(-0.006),
        Nbinsy = cms.int32(24),
        ymax = cms.double(0.006),                      
        ymin = cms.double(-0.006)
        ),

# Sim Vertex RZ
    TH2SimVtx_RZ = cms.PSet(
        Nbinsx = cms.int32(30),
        xmax = cms.double(20.0),                      
        xmin = cms.double(-20.0),
        Nbinsy = cms.int32(30),
        ymax = cms.double(0.006),                      
        ymin = cms.double(0.0)
        ),

# CW vs. TPart Eta
    TH1TPart_Eta_CW = cms.PSet(
        Nbinsx = cms.int32(45),
        xmax = cms.double(3.0),                      
        xmin = cms.double(-3.0)
        ),

# Stub modules vs. TPart Eta
    TH1TPart_Eta_PS2S = cms.PSet(
        Nbinsx = cms.int32(45),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(-3.1416)
        ),

# Cluster PID vs. Stack member
    TH2Cluster_PID = cms.PSet(
        Nbinsx = cms.int32(501),
        xmax = cms.double(250.5),                      
        xmin = cms.double(-250.5),
        Nbinsy = cms.int32(2),
        ymax = cms.double(1.5),                      
        ymin = cms.double(-0.5)
        ),

# Stub PID
    TH1Stub_PID = cms.PSet(
        Nbinsx = cms.int32(501),
        xmax = cms.double(250.5),                      
        xmin = cms.double(-250.5),
        ),   

# Track Chi2 vs TPart Eta
    TH2Track_Chi2 = cms.PSet(
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(100),
        ymax = cms.double(50),
        ymin = cms.double(0)
        ),

# Track Chi2/ndf vs TPart Eta
    TH2Track_Chi2Red = cms.PSet(
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(100),
        ymax = cms.double(10),
        ymin = cms.double(0)
        ),


# Stub InvPt vs. TPart InvpT    
    TH2Stub_InvPt = cms.PSet(
        Nbinsx = cms.int32(200),
        xmax = cms.double(0.0),                      
        xmin = cms.double(1.0),
        Nbinsy = cms.int32(200),
        ymax = cms.double(0.0),                      
        ymin = cms.double(1.0)
        ),

# Stub Pt vs. TPart pT    
    TH2Stub_Pt = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(50.0),                      
        xmin = cms.double(0.0),
        Nbinsy = cms.int32(100),
        ymax = cms.double(50.0),                      
        ymin = cms.double(0.0)
        ),

# Stub Eta vs. TPart Eta
    TH2Stub_Eta = cms.PSet(
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(180),
        ymax = cms.double(3.1416),                      
        ymin = cms.double(-3.1416)
        ),

# Stub Phi vs. TPart Phi
    TH2Stub_Phi = cms.PSet(
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(180),
        ymax = cms.double(3.1416),                      
        ymin = cms.double(-3.1416)
        ),
    
# Stub InvPtRes vs. TPart EtaRes
    TH2Stub_InvPtRes = cms.PSet(
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(100),
        ymax = cms.double(2.0),                      
        ymin = cms.double(-2.0)
        ),
    
# Stub PtRes vs. TPart EtaRes
    TH2Stub_PtRes = cms.PSet(
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(100),
        ymax = cms.double(40.0),                      
        ymin = cms.double(-40.0)
        ),

# Stub EtaRes vs. TPart Eta
    TH2Stub_EtaRes = cms.PSet(
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(100),
        ymax = cms.double(2.0),                      
        ymin = cms.double(-2.0)
        ),

# Stub PhiRes vs. TPart Eta
    TH2Stub_PhiRes = cms.PSet(
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),                      
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(100),
        ymax = cms.double(0.5),                      
        ymin = cms.double(-0.5)
        ),

# Stub Width vs. TPart InvPt
    TH2Stub_W_InvPt = cms.PSet(
        Nbinsx = cms.int32(200),
        xmax = cms.double(0.8),                      
        xmin = cms.double(0.0),
        Nbinsy = cms.int32(29),
        ymax = cms.double(7.25),                      
        ymin = cms.double(-7.25)
        ),

# Stub Width vs. TPart Pt
    TH2Stub_W_Pt = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(50.0),                      
        xmin = cms.double(0.0),
        Nbinsy = cms.int32(29),
        ymax = cms.double(7.25),                      
        ymin = cms.double(-7.25)
        ),

# Track Pt vs TPart Pt
    TH2TTTrack_Sim_Pt = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(100),
        xmin = cms.double(0),
        Nbinsy = cms.int32(100),
        ymax = cms.double(100),
        ymin = cms.double(0)
        ),

# Track PtRes vs TPart Eta
    TH2TTTrack_Sim_PtRes = cms.PSet(
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(200),
        ymax = cms.double(4.0),
        ymin = cms.double(-4.0)
        ),

# Track InvPt vs TPart InvPt
    TH2TTTrack_Sim_InvPt = cms.PSet(
        Nbinsx = cms.int32(150),
        xmax = cms.double(0.6),
        xmin = cms.double(0),
        Nbinsy = cms.int32(150),
        ymax = cms.double(0.6),
        ymin = cms.double(0)
        ),

# Track InvPtRes vs TPart Eta
    TH2TTTrack_Sim_InvPtRes = cms.PSet(
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(100),
        ymax = cms.double(1.0),
        ymin = cms.double(-1.0)
        ),

# Track Phi vs TPart Phi
    TH2TTTrack_Sim_Phi = cms.PSet(
        Nbinsx = cms.int32(90),
        xmax = cms.double(3.1416),
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(90),
        ymax = cms.double(3.1416),
        ymin = cms.double(-3.1416)
        ),

# Track PhiRes vs TPart Eta
    TH2TTTrack_Sim_PhiRes = cms.PSet(
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(100),
        ymax = cms.double(0.5),
        ymin = cms.double(-0.5)
        ),

# Track Eta vs TPart Eta
    TH2TTTrack_Sim_Eta = cms.PSet(
        Nbinsx = cms.int32(90),
        xmax = cms.double(3.1416),
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(90),
        ymax = cms.double(3.1416),
        ymin = cms.double(-3.1416)
        ),

# Track EtaRes vs TPart Eta
    TH2TTTrack_Sim_EtaRes = cms.PSet(
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(100),
        ymax = cms.double(0.5),
        ymin = cms.double(-0.5)
        ),

# Track Vtx vs TPart Vtx
    TH2TTTrack_Sim_Vtx = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(25),
        xmin = cms.double(-25),
        Nbinsy = cms.int32(100),
        ymax = cms.double(25),
        ymin = cms.double(-25)
        ),

# Track VtxRes vs TPart Eta
    TH2TTTrack_Sim_VtxRes = cms.PSet(
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(100),
        ymax = cms.double(5),
        ymin = cms.double(-5)
        ),



)
