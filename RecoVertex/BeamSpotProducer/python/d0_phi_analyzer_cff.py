import FWCore.ParameterSet.Config as cms

d0_phi_analyzer = cms.EDAnalyzer("BeamSpotAnalyzer",
    BSAnalyzerParameters = cms.PSet(
        RunAllFitters = cms.bool(False),
        WriteToDB = cms.bool(False),
        MaximumNtracks = cms.int32(1000),
        TrackCollection = cms.untracked.string('generalTracks'),
        InputBeamWidth = cms.untracked.double(-1.0),
        MinimumPt = cms.double(2.0)
    ),
    OutputFileName = cms.untracked.string('analyze_d0_phi.root')
)


