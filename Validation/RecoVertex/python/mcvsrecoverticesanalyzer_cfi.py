import FWCore.ParameterSet.Config as cms

mcvsrecoverticesanalyzer = cms.EDAnalyzer("MCvsRecoVerticesAnalyzer",
                                          pileupSummaryCollection = cms.InputTag("addPileupInfo"),
                                          mcTruthCollection = cms.InputTag("generatorSmeared"),
                                          pvCollection = cms.InputTag("offlinePrimaryVertices"),
                                          useWeight = cms.bool(False),
                                          useVisibleVertices = cms.bool(False),
                                          weightProduct = cms.InputTag("mcvertexweight")
                           )

