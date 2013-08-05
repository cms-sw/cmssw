import FWCore.ParameterSet.Config as cms

bspvanalyzer = cms.EDAnalyzer('BSvsPVAnalyzer',
                              pvCollection = cms.InputTag("offlinePrimaryVertices"),
                              bsCollection = cms.InputTag("offlineBeamSpot"),
                              firstOnly = cms.untracked.bool(False),
                              bspvHistogramMakerPSet = cms.PSet(
                                 useSlope = cms.bool(True),
                                 trueOnly = cms.untracked.bool(True),
                                 maxLSBeforeRebin = cms.uint32(100),
                                 histoParameters = cms.untracked.PSet(
                                     nBinX = cms.untracked.uint32(200), xMin=cms.untracked.double(-0.1), xMax=cms.untracked.double(0.1),
                                     nBinY = cms.untracked.uint32(200), yMin=cms.untracked.double(-0.1), yMax=cms.untracked.double(0.1),
                                     nBinZ = cms.untracked.uint32(200), zMin=cms.untracked.double(-30.), zMax=cms.untracked.double(30.),
                                     nBinZProfile = cms.untracked.uint32(60), zMinProfile=cms.untracked.double(-30.), zMaxProfile=cms.untracked.double(30.)
                                 )
                              )
                              )

