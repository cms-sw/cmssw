import FWCore.ParameterSet.Config as cms

d0_phi_analyzer_pixelless = cms.EDAnalyzer("BeamSpotAnalyzer",
    BSAnalyzerParameters = cms.PSet(
        RunAllFitters = cms.bool(False), ## False: run only default fitter
        WriteToDB = cms.bool(False), ## do not write results to DB
    ),
    BeamFitter = cms.PSet(
	Debug = cms.untracked.bool(False),
	TrackCollection = cms.untracked.InputTag('ctfPixelLess'),
	IsMuonCollection = cms.untracked.bool(False),
        WriteAscii = cms.untracked.bool(True),
	AsciiFileName = cms.untracked.string('BeamFit.txt'),
        SaveNtuple = cms.untracked.bool(False),
	SaveFitResults = cms.untracked.bool(False),
        OutputFileName = cms.untracked.string('analyze_d0_phi.root'),
	MinimumPt = cms.untracked.double(1.2),
	MaximumEta = cms.untracked.double(2.4),
	MaximumImpactParameter = cms.untracked.double(5),
	MaximumZ = cms.untracked.double(150),
	MinimumTotalLayers = cms.untracked.int32(8),
	MinimumPixelLayers = cms.untracked.int32(0),
	MaximumNormChi2 = cms.untracked.double(5.0),
	TrackAlgorithm = cms.untracked.vstring(), ## ctf,rs,cosmics,iter0,iter1...; for all algos, leave it blank
	TrackQuality = cms.untracked.vstring(), ## loose, tight, highPurity...; for all qualities, leave it blank
        InputBeamWidth = cms.untracked.double(-1.0), ## if -1 use the value calculated by the analyzer
	FractionOfFittedTrks = cms.untracked.double(0.5),
	MinimumInputTracks = cms.untracked.int32(100)
     )
)

