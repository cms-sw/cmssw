import FWCore.ParameterSet.Config as cms


from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
gemCosmicMuonStandSim = DQMEDAnalyzer('GEMCosmicMuonStandSim',
    simHitToken = cms.InputTag('g4SimHits','MuonGEMHits'),
    recHitToken = cms.InputTag('gemRecHits'),
)
