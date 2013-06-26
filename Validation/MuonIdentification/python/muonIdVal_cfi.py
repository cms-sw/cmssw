import FWCore.ParameterSet.Config as cms

muonIdVal = cms.EDAnalyzer("MuonIdVal",
    inputMuonCollection           = cms.InputTag("muons"),
    inputDTRecSegment4DCollection = cms.InputTag("dt4DSegments"),
    inputCSCSegmentCollection     = cms.InputTag("cscSegments"),
    inputMuonTimeExtraValueMap    = cms.InputTag("muons"),
    inputMuonCosmicCompatibilityValueMap = cms.InputTag("muons","cosmicsVeto"),
    inputMuonShowerInformationValueMap = cms.InputTag("muons","muonShowerInformation"), 
    useTrackerMuons               = cms.untracked.bool(True),
    useGlobalMuons                = cms.untracked.bool(True),
    useTrackerMuonsNotGlobalMuons = cms.untracked.bool(True),
    useGlobalMuonsNotTrackerMuons = cms.untracked.bool(True),
    makeEnergyPlots               = cms.untracked.bool(True),
    makeTimePlots                 = cms.untracked.bool(True),
    make2DPlots                   = cms.untracked.bool(False),
    makeAllChamberPlots           = cms.untracked.bool(False),
    makeCosmicCompatibilityPlots  = cms.untracked.bool(True),
    makeShowerInformationPlots    = cms.untracked.bool(True),
    baseFolder                    = cms.untracked.string("Muons/MuonIdentificationV")
)
