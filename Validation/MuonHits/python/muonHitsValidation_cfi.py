import FWCore.ParameterSet.Config as cms

# SimHits Validation Analyzer after Simulation
validSimHit = cms.EDAnalyzer("MuonSimHitsValidAnalyzer",
    # Name of the root file which will contain the histos
    DT_outputFile = cms.untracked.string('DTSimHitsPlots_200pre4.root'),
    Name = cms.untracked.string('MuonSimHitsValidAnalyzer'),
    RPCHitsSrc = cms.InputTag("g4SimHits","MuonRPCHits"),
    Verbosity = cms.untracked.int32(0), ## verbosity inclusive.

    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    DTHitsSrc = cms.InputTag("g4SimHits","MuonDTHits"),
    CSCHitsSrc = cms.InputTag("g4SimHits","MuonCSCHits"),
    # 0 provides no output
    # 1 provides basic output
    # 2 provides output of the fill step
    # 3 provides output of the store step
    Label = cms.string('Hits')
)


