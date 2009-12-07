import FWCore.ParameterSet.Config as cms

d0_phi_analyzer = cms.EDAnalyzer("BeamSpotAnalyzer",
    BSAnalyzerParameters = cms.PSet(
        RunAllFitters = cms.bool(False), ## run only default fitter

        WriteToDB = cms.bool(False), ## do not write results to DB

        MaximumNtracks = cms.int32(1000), ## disable for the moment

        TrackCollection = cms.untracked.string('generalTracks'),
        InputBeamWidth = cms.untracked.double(-1.0), ## if -1 use the value calculated by the analyzer

        MinimumPt = cms.double(2.0) ## Gev/c

    ),
    OutputFileName = cms.untracked.string('analyze_d0_phi.root')
)


