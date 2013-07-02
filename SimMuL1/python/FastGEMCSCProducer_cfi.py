import FWCore.ParameterSet.Config as cms

from GEMCode.GEMValidation.simTrackMatching_cfi import SimTrackMatching

stm = SimTrackMatching.clone()

stm.useCSCChamberTypes = cms.untracked.vint32( 2, 5 )
stm.minBXCSCComp = 0
stm.maxBXCSCComp = 16
stm.minBXCSCWire = 0
stm.maxBXCSCWire = 16
stm.minBXCLCT = 0
stm.maxBXCLCT = 16
stm.minBXALCT = 0
stm.maxBXALCT = 16
stm.minBXLCT = 0
stm.maxBXLCT = 16
# turn off all matchers except SimHitMatcher
stm.gemDigiInput = cms.untracked.InputTag("")
stm.cscWireDigiInput = cms.untracked.InputTag("")
stm.cscLCTInput = cms.untracked.InputTag("")
 


FastGEMCSCProducer = cms.EDProducer("FastGEMCSCProducer",
    verbose = cms.untracked.int32(0),
    simInputLabel = cms.untracked.string("g4SimHits"),
    lctInput = cms.untracked.InputTag("simCscTriggerPrimitiveDigis", "MPCSORTED"),
    productInstanceName = cms.untracked.string("FastGEM"),
    minPt = cms.untracked.double(4.5),
    minEta = cms.untracked.double(1.55),
    maxEta = cms.untracked.double(2.4),
    usePropagatedDPhi = cms.untracked.bool(True),
    # index-to-chamber-type: 0:dummy, 1:ME1/a, 2:ME1/b, 3:ME1/2, 4:ME1/3, 5: ME2/1, ...
    #zOddGEM = cms.vdouble(-1., -1., 569.7, -1., -1., 798.3, -1., -1., -1., -1., -1.), # comparable to ME11
    #zEvenGEM = cms.vdouble(-1., -1., 567.6, -1., -1., 796.2, -1., -1., -1., -1., -1.),# comparable to ME11
    zOddGEM = cms.vdouble(-1., -1., 569.7, -1., -1., 792.3, -1., -1., -1., -1., -1.), # + 6cm lever arm
    zEvenGEM = cms.vdouble(-1., -1., 567.6, -1., -1., 790.2, -1., -1., -1., -1., -1.),# + 6cm lever arm
    # half-strip phi-pitch flat smearing
    phiSmearCSC = cms.vdouble(-1., -1., 0.00148, -1., -1., 0.00233, -1., -1., -1., -1., -1.),
    # trigger pad phi-pitch flat smearing
    phiSmearGEM = cms.vdouble(-1., -1., 0.00190, -1., -1., 2*0.00190, -1., -1., -1., -1., -1.),
    simTrackMatching = stm
)
