import FWCore.ParameterSet.Config as cms


from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
gemCosmicMuonStandEfficiency = DQMEDAnalyzer('GEMCosmicMuonStandEfficiency',
    outsideInTracks = cms.InputTag('GEMCosmicMuon'),
    insideOutTracks = cms.InputTag('GEMCosmicMuonInSide'),
)
