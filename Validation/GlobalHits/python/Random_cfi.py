import FWCore.ParameterSet.Config as cms

RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits = cms.untracked.uint32(11),
        ecalUnsuppressedDigis = cms.untracked.uint32(1234567),
        muonCSCDigis = cms.untracked.uint32(11223344),
        mix = cms.untracked.uint32(12345),
        muonDTDigis = cms.untracked.uint32(1234567),
        VtxSmeared = cms.untracked.uint32(98765432),
        siPixelDigis = cms.untracked.uint32(1234567),
        siStripDigis = cms.untracked.uint32(1234567),
        hcalDigis = cms.untracked.uint32(11223344),
        muonRPCDigis = cms.untracked.uint32(1234567)
    ),
    sourceSeed = cms.untracked.uint32(123456789)
)


