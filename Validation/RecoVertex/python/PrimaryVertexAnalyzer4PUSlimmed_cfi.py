import FWCore.ParameterSet.Config as cms

selectedOfflinePrimaryVertices = cms.EDFilter("VertexSelector",
                                               src = cms.InputTag('offlinePrimaryVertices'),
                                               cut = cms.string("isValid & ndof > 4 & tracksSize > 0 & abs(z) <= 24 & abs(position.Rho) <= 2."),
                                               filter = cms.bool(False)
)

selectedOfflinePrimaryVerticesWithBS = selectedOfflinePrimaryVertices.clone()
selectedOfflinePrimaryVerticesWithBS.src = cms.InputTag('offlinePrimaryVerticesWithBS')

selectedPixelVertices = selectedOfflinePrimaryVertices.clone()
selectedPixelVertices.src = cms.InputTag('pixelVertices')

vertexAnalysis = cms.EDAnalyzer("PrimaryVertexAnalyzer4PUSlimmed",
                                simG4 = cms.InputTag("g4SimHits"),
                                use_TP_associator = cms.untracked.bool(False),
                                verbose = cms.untracked.bool(False),
                                sigma_z_match = cms.untracked.double(3.0),
                                recoTrackProducer = cms.untracked.string("generalTracks"),
                                vertexRecoCollections = cms.VInputTag("offlinePrimaryVertices",
                                                                      "offlinePrimaryVerticesWithBS",
                                                                      "pixelVertices",
                                                                      "selectedOfflinePrimaryVertices",
                                                                      "selectedOfflinePrimaryVerticesWithBS",
                                                                      "selectedPixelVertices"),
)

vertexAnalysisSequence = cms.Sequence(cms.ignore(selectedOfflinePrimaryVertices)
                                      * cms.ignore(selectedOfflinePrimaryVerticesWithBS)
                                      * cms.ignore(selectedPixelVertices)
                                      * vertexAnalysis
)
