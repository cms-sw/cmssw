import FWCore.ParameterSet.Config as cms
import math

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
OuterTrackerMonitorTrackingParticles = DQMEDAnalyzer('OuterTrackerMonitorTrackingParticles',
    TopFolderName = cms.string('SiOuterTrackerV'),
    trackingParticleToken = cms.InputTag("mix","MergedTrackTruth"), #tracking particles
    StubInputTag = cms.InputTag("TTStubsFromPhase2TrackerDigis","StubAccepted"), #stubs
    TTTracksTag       = cms.InputTag("TTTracksFromTracklet", "Level1TTTracks"), #tracks (currently from tracklet)
    MCTruthStubInputTag = cms.InputTag("TTStubAssociatorFromPixelDigis", "StubAccepted"), #truth stub associator
    MCTruthTrackInputTag = cms.InputTag("TTTrackAssociatorFromPixelDigis", "Level1TTTracks"), #truth track associator
    MCTruthClusterInputTag = cms.InputTag("TTClusterAssociatorFromPixelDigis", "ClusterAccepted"), #truth cluster associator
    L1Tk_nPar = cms.int32(4),           # use 4 or 5-parameter L1 track fit ??
    L1Tk_minNStub = cms.int32(4),       # L1 tracks with >= 4 stubs
    L1Tk_maxChi2 = cms.double(400.0),   # L1 tracks with Chi2 <= X
    L1Tk_maxChi2dof = cms.double(100.0),# L1 tracks with Chi2 <= X
    TP_minNStub = cms.int32(4),      # require TP to have >= X number of stubs associated with it
    TP_minPt = cms.double(2.0),      # only save TPs with pt > X GeV
    TP_maxPt = cms.double(1000.0),   # only save TPs with pt < X GeV
    TP_maxEta = cms.double(2.4),     # only save TPs with |eta| < X
    TP_maxVtxZ = cms.double(30.0),     # only save TPs with |z0| < X cm
    TP_select_eventid = cms.int32(0),# if zero, only look at TPs from primary interaction, else, include TPs from pileup

# tracking particles vs eta
    TH1TrackParts_Eta = cms.PSet(
        Nbinsx = cms.int32(45),
        xmax = cms.double(3),
        xmin = cms.double(-3)
        ),

# tracking particles vs phi
    TH1TrackParts_Phi = cms.PSet(
        Nbinsx = cms.int32(60),
        xmax = cms.double(math.pi),
        xmin = cms.double(-math.pi)
        ),

# tracking particles vs pT
    TH1TrackParts_Pt = cms.PSet(
        Nbinsx = cms.int32(45),
        xmax = cms.double(100),
        xmin = cms.double(0)
        ),

#Chi2 of the track
    TH1_Track_Chi2 = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(50),
        xmin = cms.double(0)
        ),

#Chi2R of the track
    TH1_Track_Chi2R = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(10),
        xmin = cms.double(0)
        ),

# tracking particles vs pT_relative
    TH1Res_ptRel = cms.PSet(
        Nbinsx = cms.int32(200),
        xmax = cms.double(0.5),
        xmin = cms.double(-0.5)
        ),

# tracking particles vs pT (for efficiency)
    TH1Effic_pt = cms.PSet(
        Nbinsx = cms.int32(50),
        xmax = cms.double(100),
        xmin = cms.double(0)
        ),

# tracking particles vs pT (for efficiency)
    TH1Effic_pt_zoom = cms.PSet(
        Nbinsx = cms.int32(50),
        xmax = cms.double(10),
        xmin = cms.double(0)
        ),

# tracking particles vs eta (for efficiency)
    TH1Effic_eta = cms.PSet(
        Nbinsx = cms.int32(50),
        xmax = cms.double(2.5),
        xmin = cms.double(-2.5)
        ),

# tracking particles vs d0 (for efficiency)
    TH1Effic_d0 = cms.PSet(
        Nbinsx = cms.int32(50),
        xmax = cms.double(2),
        xmin = cms.double(-2)
        ),

# tracking particles vs VtxR/vxy (for efficiency)
    TH1Effic_VtxR = cms.PSet(
        Nbinsx = cms.int32(50),
        xmax = cms.double(5),
        xmin = cms.double(-5)
        ),

# tracking particles vs z0 (for efficiency)
    TH1Effic_VtxZ = cms.PSet(
        Nbinsx = cms.int32(50),
        xmax = cms.double(30),
        xmin = cms.double(-30)
        ),

# tracking particles vs relative pT (for resolution plots)
    TH1Res_pt = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(0.2),
        xmin = cms.double(-0.2)
        ),

# tracking particles vs eta (for resolution)
    TH1Res_eta = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(0.01),
        xmin = cms.double(-0.01)
        ),

# tracking particles vs phi (for resolution)
    TH1Res_phi = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(0.01),
        xmin = cms.double(-0.01)
        ),

# tracking particles vs z0 (for resolution)
    TH1Res_VtxZ = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(1.0),
        xmin = cms.double(-1.0)
        ),

# tracking particles vs d0 (for resolution)
    TH1Res_d0 = cms.PSet(
        Nbinsx = cms.int32(100),
        xmax = cms.double(0.05),
        xmin = cms.double(-0.05)
        ),
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(OuterTrackerMonitorTrackingParticles, trackingParticleToken = "mixData:MergedTrackTruth")
