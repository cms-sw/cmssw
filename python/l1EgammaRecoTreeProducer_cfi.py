import FWCore.ParameterSet.Config as cms

l1EgammaRecoTreeProducer = cms.EDAnalyzer(
    "L1EgammaRecoTreeProducer",
    ebSCTag = cms.untracked.InputTag("hybridSuperClusters"),
    eeSCTag = cms.untracked.InputTag("multi5x5SuperClusters")
)

