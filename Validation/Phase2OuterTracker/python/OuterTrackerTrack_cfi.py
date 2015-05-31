import FWCore.ParameterSet.Config as cms

OuterTrackerTrack = cms.EDAnalyzer('OuterTrackerTrack',
    
    TopFolderName  = cms.string('Phase2OuterTrackerV'),
    TTTracks       = cms.InputTag("TTTracksFromPixelDigis", "Level1TTTracks"),
    TTTrackMCTruth = cms.InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks"),
    HQDelim        = cms.int32(4),
    verbosePlots   = cms.untracked.bool(False),
    
    
# Number of Stubs in Track
    TH1TTTrack_NStubs = cms.PSet(
        Nbinsx = cms.int32(11),
        xmax = cms.double(10.5),
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
        Nbinsx = cms.int32(100),
        xmax = cms.double(100),
        xmin = cms.double(0)
        ),

# Track Eta
    TH1TTTrack_Eta = cms.PSet(
        Nbinsx = cms.int32(90),
        xmax = cms.double(3.1416),
        xmin = cms.double(-3.1416)
        ),

# Track Phi
    TH1TTTrack_Phi = cms.PSet(
        Nbinsx = cms.int32(90),
        xmax = cms.double(3.1416),
        xmin = cms.double(-3.1416)
        ),

#Track Vertex Position in z
    TH1TTTrack_VtxZ0 = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(25),                      
        xmin = cms.double(-25)
        ),

#Track Chi2
    TH1TTTrack_Chi2 = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(50),                      
        xmin = cms.double(0)
        ),

#Track Chi2/ndf
    TH1TTTrack_Chi2Red = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(10),                      
        xmin = cms.double(0)
        ),

# Track Chi2 vs NStubs
    TH2TTTrack_Chi2 = cms.PSet(
        Nbinsx = cms.int32(11),
        xmax = cms.double(10.5),
        xmin = cms.double(-0.5),
        Nbinsy = cms.int32(100),
        ymax = cms.double(50),
        ymin = cms.double(0)
        ),

# Track Chi2/ndf vs NStubs
    TH2TTTrack_Chi2Red = cms.PSet(
        Nbinsx = cms.int32(11),
        xmax = cms.double(10.5),
        xmin = cms.double(-0.5),
        Nbinsy = cms.int32(100),
        ymax = cms.double(10),
        ymin = cms.double(0)
        ),



)
