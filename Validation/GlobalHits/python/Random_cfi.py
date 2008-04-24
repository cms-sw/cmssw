import FWCore.ParameterSet.Config as cms

RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(11),
        simEcalUnsuppressedDigis = cms.untracked.uint32(1234567),
        simMuonCSCDigis = cms.untracked.uint32(11223344),
        mix = cms.untracked.uint32(12345),
        simMuonDTDigis = cms.untracked.uint32(1234567),
        VtxSmeared = cms.untracked.uint32(98765432),
        simSiPixelDigis = cms.untracked.uint32(1234567),
        simSiStripDigis = cms.untracked.uint32(1234567),
        simHcalDigis = cms.untracked.uint32(11223344),
        simMuonRPCDigis = cms.untracked.uint32(1234567)
    ),
    sourceSeed = cms.untracked.uint32(123456789)
)


