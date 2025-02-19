import FWCore.ParameterSet.Config as cms

globalhitprodhiststrip = cms.EDAnalyzer("GlobalHitsProdHistStripper",
    # 1 provides basic output
    # 2 provides output of the fill step + 1
    # 3 provides output of the store step + 2
    OutputFile = cms.string('GlobalHitsHistograms.root'),
    Name = cms.untracked.string('GlobalHitsHistogrammer'),
    Verbosity = cms.untracked.int32(0), ## 0 provides no output

    # as of 110p2, needs to be 1. Anything ealier should be 0.
    VtxUnit = cms.untracked.int32(1),
    Frequency = cms.untracked.int32(1),
    DoOutput = cms.bool(False),
    # 1 assumes cm in SimVertex
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    )
)


