import FWCore.ParameterSet.Config as cms

beamspotanalyzer = cms.EDAnalyzer('AnotherBeamSpotAnalyzer',
                                         bsCollection = cms.InputTag("offlineBeamSpot"),
                                         bsHistogramMakerPSet = cms.PSet(
                                               histoParameters = cms.untracked.PSet(
                                                 nBinX = cms.untracked.uint32(200), xMin=cms.untracked.double(-1.), xMax=cms.untracked.double(1.),
                                                 nBinY = cms.untracked.uint32(200), yMin=cms.untracked.double(-1.), yMax=cms.untracked.double(1.),
                                                 nBinZ = cms.untracked.uint32(200), zMin=cms.untracked.double(-20.), zMax=cms.untracked.double(20.),
                                                 nBinSigmaX = cms.untracked.uint32(200), sigmaXMin=cms.untracked.double(0.), sigmaXMax=cms.untracked.double(0.025),
                                                 nBinSigmaY = cms.untracked.uint32(200), sigmaYMin=cms.untracked.double(0.), sigmaYMax=cms.untracked.double(0.025),
                                                 nBinSigmaZ = cms.untracked.uint32(200), sigmaZMin=cms.untracked.double(0.), sigmaZMax=cms.untracked.double(15.)
                                               )
                                         )
                                      )

