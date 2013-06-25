import FWCore.ParameterSet.Config as cms

from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching

SimTrackMatching.useCSCChamberTypes = cms.untracked.vint32( 2, 5 )
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
# turn off all matchers except SimHitMatcher
SimTrackMatching.gemDigiInput = cms.untracked.InputTag("")
SimTrackMatching.cscWireDigiInput = cms.untracked.InputTag("")
SimTrackMatching.cscLCTInput = cms.untracked.InputTag("")
 


FastGE21CSCProducer = cms.EDProducer("FastGE21CSCProducer",
    verbose = cms.untracked.int32(0),
    simInputLabel = cms.untracked.string("g4SimHits"),
    lctInput = cms.untracked.InputTag("simCscTriggerPrimitiveDigis", "MPCSORTED"),
    productInstanceName = cms.untracked.string("FastGE21"),
    minPt = cms.untracked.double(4.5),
    minEta = cms.untracked.double(1.55),
    maxEta = cms.untracked.double(2.4),
    usePropagatedDPhi_ = cms.untracked.bool(True),
    # index-to-chamber-type: 0:dummy, 1:ME1/a, 2:ME1/b, 3:ME1/2, 4:ME1/3, 5: ME2/1, ...
    zOddGE21 = cms.vdouble(-1., -1., 569.7, -1., -1., 798.3, -1., -1., -1., -1., -1.),
    zEvenGE21 = cms.vdouble(-1., -1., 567.6, -1., -1., 796.2, -1., -1., -1., -1., -1.),
    # half-strip phi-pitch flat smearing
    phiSmearCSC = cms.vdouble(-1., -1., 0.00148, -1., -1., 0.00233, -1., -1., -1., -1., -1.),
    # trigger pad phi-pitch flat smearing
    phiSmearGEM = cms.vdouble(-1., -1., 0.00190, -1., -1., 0.00190, -1., -1., -1., -1., -1.),
    simTrackMatching = SimTrackMatching
)
