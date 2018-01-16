import FWCore.ParameterSet.Config as cms

globalhithistogram = DQMStep1Module('GlobalHitsHistogrammer',
    # 1 provides basic output
    # 2 provides output of the fill step + 1
    # 3 provides output of the store step + 2
    OutputFile = cms.string('GlobalHitsHistograms.root'),
    Name = cms.untracked.string('GlobalHitsHistogrammer'),
    GlobalHitSrc = cms.InputTag("globalhits","GlobalHits"),
    Verbosity = cms.untracked.int32(0), ## 0 provides no output

    # as of 110p2, needs to be 1. Anything ealier should be 0.
    VtxUnit = cms.untracked.int32(1),
    Frequency = cms.untracked.int32(50),
    DoOutput = cms.bool(False),
    # 1 assumes cm in SimVertex
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    )
)


