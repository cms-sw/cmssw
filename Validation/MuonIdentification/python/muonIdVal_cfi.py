import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
muonIdVal = DQMEDAnalyzer('MuonIdVal',
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

# fastsim has no cosmic muon veto in place
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(muonIdVal, makeCosmicCompatibilityPlots = False)
