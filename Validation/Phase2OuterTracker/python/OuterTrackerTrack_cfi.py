import FWCore.ParameterSet.Config as cms

OuterTrackerTrack = cms.EDAnalyzer('OuterTrackerTrack',
    
    TopFolderName = cms.string('Phase2OuterTrackerV'),
    
# Number of Stubs in Track
    TH1TTTrack_NStubs = cms.PSet(
        Nbinsx = cms.int32(16),
        xmax = cms.double(15.5),
        xmin = cms.double(-0.5)
        ),

# Track Stacks
    TH1TTTrack_N = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(99.5),
        xmin = cms.double(-0.5)
        ),

# Track Pt
    TH1TTTrack_Pt = cms.PSet(
        Nbinsx = cms.int32(200),
        xmax = cms.double(100),
        xmin = cms.double(0)
        ),

# Track Eta
    TH1TTTrack_Eta = cms.PSet(
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),
        xmin = cms.double(-3.1416)
        ),

# Track Phi
    TH1TTTrack_Phi = cms.PSet(
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),
        xmin = cms.double(-3.1416)
        ),

# Track Chi2
    TH2TTTrack_Chi2 = cms.PSet(
        Nbinsx = cms.int32(20),
        xmax = cms.double(19.5),
        xmin = cms.double(-0.5),
        Nbinsy = cms.int32(200),
        ymax = cms.double(50),
        ymin = cms.double(0)
        ),

# Track Chi2/ndf
    TH2TTTrack_Chi2Red = cms.PSet(
        Nbinsx = cms.int32(20),
        xmax = cms.double(19.5),
        xmin = cms.double(-0.5),
        Nbinsy = cms.int32(200),
        ymax = cms.double(10),
        ymin = cms.double(0)
        ),

# Track Pt vs TPart Pt
    TH2TTTrack_Sim_Pt = cms.PSet(
        Nbinsx = cms.int32(200),
        xmax = cms.double(100),
        xmin = cms.double(0),
        Nbinsy = cms.int32(200),
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
        Nbinsx = cms.int32(200),
        xmax = cms.double(0.8),
        xmin = cms.double(0),
        Nbinsy = cms.int32(200),
        ymax = cms.double(0.8),
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
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(180),
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
        Nbinsx = cms.int32(180),
        xmax = cms.double(3.1416),
        xmin = cms.double(-3.1416),
        Nbinsy = cms.int32(180),
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
        Nbinsx = cms.int32(180),
        xmax = cms.double(30),
        xmin = cms.double(-30),
        Nbinsy = cms.int32(180),
        ymax = cms.double(30),
        ymin = cms.double(-30)
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
