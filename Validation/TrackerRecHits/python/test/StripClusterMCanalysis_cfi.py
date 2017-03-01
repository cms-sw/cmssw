import FWCore.ParameterSet.Config as cms

StripClusterMCanalysis = cms.EDAnalyzer('StripClusterMCanalysis',
                                    beamSpot = cms.InputTag("offlineBeamSpot"),
                                    primaryVertex = cms.InputTag("firstStepPrimaryVertices"),
                                    stripClusters = cms.InputTag("siStripClusters"),
                                    stripSimLinks = cms.InputTag("simSiStripDigis"),
                                    printOut = cms.untracked.int32(0),
                                    ROUList = cms.vstring(
                                      'TrackerHitsTIBLowTof',  # hard-scatter (prompt) collection, module label g4simHits
                                      'TrackerHitsTIBHighTof',
                                      'TrackerHitsTIDLowTof',
                                      'TrackerHitsTIDHighTof',
                                      'TrackerHitsTOBLowTof',
                                      'TrackerHitsTOBHighTof',
                                      'TrackerHitsTECLowTof',
                                      'TrackerHitsTECHighTof'
                                      )
                                   )
