import FWCore.ParameterSet.Config as cms

# SimHits Validation Analyzer after Simulation
validSimHit = cms.EDAnalyzer("MuonSimHitsValidAnalyzer",
    # Name of the root file which will contain the histos
    DT_outputFile = cms.untracked.string(''),
    Name = cms.untracked.string('MuonSimHitsValidAnalyzer'),
    RPCHitsSrc = cms.InputTag("g4SimHits","MuonRPCHits"),
    Verbosity = cms.untracked.int32(0),
    ProvenanceLookup = cms.PSet(
        PrintProvenanceInfo = cms.untracked.bool(False),
        GetAllProvenances = cms.untracked.bool(False)
    ),
    DTHitsSrc = cms.InputTag("g4SimHits","MuonDTHits"),
    CSCHitsSrc = cms.InputTag("g4SimHits","MuonCSCHits"),
    Label = cms.string('Hits')
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    validSimHit.DTHitsSrc = cms.InputTag("MuonSimHits","MuonDTHits")
    validSimHit.CSCHitsSrc = cms.InputTag("MuonSimHits","MuonCSCHits")
    validSimHit.RPCHitsSrc = cms.InputTag("MuonSimHits","MuonRPCHits")
    
