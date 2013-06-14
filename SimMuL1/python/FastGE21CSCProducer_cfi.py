import FWCore.ParameterSet.Config as cms

from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching

SimTrackMatching.usedChamberTypesCSC = cms.untracked.vint32( 5, )
SimTrackMatching.minBXCSCComp = 0
SimTrackMatching.maxBXCSCComp = 16
SimTrackMatching.minBXCSCWire = 0
SimTrackMatching.maxBXCSCWire = 16
SimTrackMatching.minBXCLCT = 0
SimTrackMatching.maxBXCLCT = 16
SimTrackMatching.minBXALCT = 0
SimTrackMatching.maxBXALCT = 16
SimTrackMatching.minBXLCT = 0
SimTrackMatching.maxBXLCT = 16


FastGE21CSCProducer = cms.EDProducer("FastGE21CSCProducer",
    verbose = cms.untracked.int32(0),
    simInputLabel = cms.untracked.string("g4SimHits"),
    lctInput = cms.untracked.InputTag("simCscTriggerPrimitiveDigis", "MPCSORTED"),
    productInstanceName = cms.untracked.string("FastGE21"),
    minPt = cms.untracked.double(4.5),
    cscType = cms.untracked.int32(5),
    zOddGE21 = cms.untracked.double(780.),
    zEvenGE21 = cms.untracked.double(775.),
    createNtuple = cms.untracked.bool(False),
    simTrackMatching = SimTrackMatching
)
