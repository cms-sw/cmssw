import FWCore.ParameterSet.Config as cms

#
# additional triggerMatching for caloTaus
#
from PhysicsTools.PatAlgos.triggerLayer0.patTrigProducer_cfi import patHLT1Tau
from PhysicsTools.PatAlgos.triggerLayer0.patTrigProducer_cfi import patHLT2TauPixel

# matches to HLT1Tau
tauTrigMatchHLT1CaloTau = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0CaloTaus"),
    matched = cms.InputTag("patHLT1Tau"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

# matches to HLT2TauPixel
tauTrigMatchHLT2CaloTauPixel = cms.EDFilter("PATTrigMatcher",
    src     = cms.InputTag("allLayer0CaloTaus"),
    matched = cms.InputTag("patHLT2TauPixel"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)
