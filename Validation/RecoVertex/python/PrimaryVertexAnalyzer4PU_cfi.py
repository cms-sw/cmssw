import FWCore.ParameterSet.Config as cms

vertexAnalysis = cms.EDAnalyzer("PrimaryVertexAnalyzer4PU",
        simG4 = cms.InputTag("g4SimHits"),
        outputFile = cms.untracked.string("pv.root"),
        verbose = cms.untracked.bool(True),
        recoTrackProducer = cms.untracked.string("generalTracks"),
        zmatch=cms.untracked.double(0.05),
        TkFilterParameters = cms.PSet(
          maxNormalizedChi2 = cms.double(5.0),
          minSiliconHits = cms.int32(7), ## hits > 7
          maxD0Significance = cms.double(5.0), ## keep most primary tracks
          minPt = cms.double(0.0), ## better for softish events
          minPixelHits = cms.int32(2) ## hits > 2
        )
)
