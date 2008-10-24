import FWCore.ParameterSet.Config as cms

muonIdVal = cms.EDAnalyzer("MuonIdVal",
    inputMuonCollection           = cms.InputTag("muons"),
    inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
    inputCSCSegmentCollection     = cms.InputTag("cscSegments"),
    useTrackerMuons               = cms.untracked.bool(True),
    useGlobalMuons                = cms.untracked.bool(True),
    makeDQMPlots                  = cms.untracked.bool(False),
    makeEnergyPlots               = cms.untracked.bool(False),
    makeIsoPlots                  = cms.untracked.bool(False),
    make2DPlots                   = cms.untracked.bool(False),
    makeAllChamberPlots           = cms.untracked.bool(False),
    baseFolder                    = cms.untracked.string("Muons/MuonIdVal"),
)
