import FWCore.ParameterSet.Config as cms

beamspotanalyzer = cms.EDAnalyzer('AnotherBeamSpotAnalyzer',
                                         bsCollection = cms.InputTag("offlineBeamSpot"),
                                         bsHistogramMakerPSet = cms.PSet(
                                               histoParameters = cms.untracked.PSet(
                                                 nBinX = cms.untracked.uint32(200), xMin=cms.untracked.double(-1.), xMax=cms.untracked.double(1.),
                                                 nBinY = cms.untracked.uint32(200), yMin=cms.untracked.double(-1.), yMax=cms.untracked.double(1.),
                                                 nBinZ = cms.untracked.uint32(200), zMin=cms.untracked.double(-20.), zMax=cms.untracked.double(20.),
                                                 nBinSigmaZ = cms.untracked.uint32(200), sigmaZMin=cms.untracked.double(0.), sigmaZMax=cms.untracked.double(15.)
                                               )
                                         )
                                      )

