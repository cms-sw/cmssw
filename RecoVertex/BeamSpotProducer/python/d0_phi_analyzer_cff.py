import FWCore.ParameterSet.Config as cms

d0_phi_analyzer = cms.EDAnalyzer("BeamSpotAnalyzer",
    BSAnalyzerParameters = cms.PSet(
        RunAllFitters = cms.bool(False), ## False: run only default fitter
        RunBeamWidthFit = cms.bool(False), 
        WriteToDB = cms.bool(False), ## do not write results to DB
        fitEveryNLumi = cms.untracked.int32( -1 ),
        resetEveryNLumi = cms.untracked.int32( -1 )
    ),
    BeamFitter = cms.PSet(
        Debug = cms.untracked.bool(False),
        TrackCollection = cms.untracked.InputTag('generalTracks'),
        IsMuonCollection = cms.untracked.bool(False),
        WriteAscii = cms.untracked.bool(True),
        AsciiFileName = cms.untracked.string('BeamFit.txt'), ## all results
        AppendRunToFileName = cms.untracked.bool(True), #runnumber will be inserted to the file name
        WriteDIPAscii = cms.untracked.bool(False),
        DIPFileName = cms.untracked.string('BeamFitDIP.txt'), ## only the last results, for DIP
        SaveNtuple = cms.untracked.bool(False),
        SavePVVertices = cms.untracked.bool(False),
        SaveFitResults = cms.untracked.bool(False),
        OutputFileName = cms.untracked.string('analyze_d0_phi.root'),
        MinimumPt = cms.untracked.double(1.2),
        MaximumEta = cms.untracked.double(2.4),
        MaximumImpactParameter = cms.untracked.double(5),
        MaximumZ = cms.untracked.double(60),
        MinimumTotalLayers = cms.untracked.int32(11),
        MinimumPixelLayers = cms.untracked.int32(3),
        MaximumNormChi2 = cms.untracked.double(2.0),
        TrackAlgorithm = cms.untracked.vstring('initialStep'), ## ctf,rs,cosmics,initialStep,lowPtTripletStep...; for all algos, leave it blank
        TrackQuality = cms.untracked.vstring(), ## loose, tight, highPurity...; for all qualities, leave it blank
        InputBeamWidth = cms.untracked.double(0.0060), ## beam width used for Trk fitter, used only when result from PV is not available
        FractionOfFittedTrks = cms.untracked.double(0.9),
        MinimumInputTracks = cms.untracked.int32(100)
     ),
     PVFitter = cms.PSet(
        Debug = cms.untracked.bool(False),
        Apply3DFit = cms.untracked.bool(False),
        VertexCollection = cms.untracked.InputTag('offlinePrimaryVertices'),
        #WriteAscii = cms.untracked.bool(True),
        #AsciiFileName = cms.untracked.string('PVFit.txt'),
        maxNrStoredVertices = cms.untracked.uint32(10000),
        minNrVerticesForFit = cms.untracked.uint32(100),
        minVertexNdf = cms.untracked.double(10.),
        maxVertexNormChi2 = cms.untracked.double(10.),
        minVertexNTracks = cms.untracked.uint32(0),
        minVertexMeanWeight = cms.untracked.double(0.5),
        maxVertexR = cms.untracked.double(2),
        maxVertexZ = cms.untracked.double(10),
        errorScale = cms.untracked.double(0.9),
        nSigmaCut = cms.untracked.double(5.),
        FitPerBunchCrossing = cms.untracked.bool(False),
        useOnlyFirstPV = cms.untracked.bool(False),
        minSumPt = cms.untracked.double(0.)
     )
)

