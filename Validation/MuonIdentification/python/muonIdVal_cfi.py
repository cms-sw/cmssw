import FWCore.ParameterSet.Config as cms

muonIdVal = cms.EDAnalyzer("MuonIdVal",
    inputMuonCollection           = cms.InputTag("muons"),
    inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
    inputCSCSegmentCollection     = cms.InputTag("cscSegments"),
    useTrackerMuons               = cms.untracked.bool(True),
    useGlobalMuons                = cms.untracked.bool(True),
    makeEnergyPlots               = cms.untracked.bool(True),
    make2DPlots                   = cms.untracked.bool(False),
    makeAllChamberPlots           = cms.untracked.bool(False),
    baseFolder                    = cms.untracked.string("Muons/MuonIdVal")
)
