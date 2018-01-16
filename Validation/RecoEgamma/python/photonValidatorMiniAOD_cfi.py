import FWCore.ParameterSet.Config as cms

photonValidationMiniAOD = DQMStep1Module('PhotonValidatorMiniAOD',
    ComponentName = cms.string('photonValidationMiniAOD'),
    outputFileName  = cms.string("output.root"),
    PhotonTag=cms.untracked.InputTag('slimmedPhotons'),
    genpartTag=cms.untracked.InputTag('prunedGenParticles'),
#
    eBin = cms.int32(100),
    eMin = cms.double(0.0),
    eMax = cms.double(500.0),
#
    etBin = cms.int32(100),
    etMax = cms.double(250.),
    etMin = cms.double(0.0),
#
    etaBin = cms.int32(100),
    etaMin = cms.double(-2.5),
    etaMax = cms.double(2.5),
#
    phiBin = cms.int32(100),
    phiMin = cms.double(-3.14),
    phiMax = cms.double(3.14),
#
    r9Bin = cms.int32(200),
    r9Min = cms.double(0.0),
    r9Max = cms.double(1.1),
#
    resBin = cms.int32(100),
    resMin = cms.double(0.7),
    resMax = cms.double(1.2)
#



)


