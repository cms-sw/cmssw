import FWCore.ParameterSet.Config as cms

#
# define additional recoSequences within pat
#

#
# caloTaus
#

## object cleaning
from PhysicsTools.PatAlgos.cleaningLayer0.caloTauCleaner_cfi import allLayer0CaloTaus

## standard sequence for caloTauCleaner
tqafLayer0Cleaners_withCaloTau = cms.Sequence(
    allLayer0CaloTaus
)

## mc matching
import PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi

## clone module(s)
caloTauMatch           = PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi.tauMatch.clone()
caloTauMatch.src       = 'allLayer0CaloTaus'

caloTauGenJetMatch     = PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi.tauGenJetMatch.clone()
caloTauGenJetMatch.src = 'allLayer0CaloTaus'

## standard sequence for mcTruthMatching
tqafLayer0MCTruth_withCaloTau = cms.Sequence(
    caloTauMatch *
    caloTauGenJetMatch
)

## trigger matching
from PhysicsTools.PatAlgos.triggerLayer0.patTrigProducer_cfi import patHLT1Tau

tauTrigMatchHLT1CaloTau = cms.EDProducer("PATTrigMatcher",
    src       = cms.InputTag("allLayer0CaloTaus"),
    matched   = cms.InputTag("patHLT1Tau"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

from PhysicsTools.PatAlgos.triggerLayer0.patTrigProducer_cfi import patHLT2TauPixel

tauTrigMatchHLT2CaloTauPixel = cms.EDProducer("PATTrigMatcher",
    src       = cms.InputTag("allLayer0CaloTaus"),
    matched   = cms.InputTag("patHLT2TauPixel"),
    maxDPtRel = cms.double(1.0),
    maxDeltaR = cms.double(0.2),
    resolveAmbiguities    = cms.bool(True),
    resolveByMatchQuality = cms.bool(False),
)

## standard sequence for triggerMatching
tqafLayer0TrigMatch_withCaloTaus = cms.Sequence(
    patHLT1Tau * tauTrigMatchHLT1CaloTau +
    patHLT2TauPixel * tauTrigMatchHLT2CaloTauPixel
)

## caloTau producer and selectors
import PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi
import PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi
import PhysicsTools.PatAlgos.selectionLayer1.tauMinFilter_cfi
import PhysicsTools.PatAlgos.selectionLayer1.tauMaxFilter_cfi

## clone module(s)
allLayer1CaloTaus      = PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi.allLayer1Taus.clone()
minLayer1CaloTaus      = PhysicsTools.PatAlgos.selectionLayer1.tauMinFilter_cfi.minLayer1Taus.clone()
maxLayer1CaloTaus      = PhysicsTools.PatAlgos.selectionLayer1.tauMaxFilter_cfi.maxLayer1Taus.clone()
selectedLayer1CaloTaus = PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi.selectedLayer1Taus.clone()

## do the proper replacements
allLayer1CaloTaus.tauSource            = 'allLayer0CaloTaus'
allLayer1CaloTaus.addTauID             = False
allLayer1CaloTaus.trigPrimMatch        = cms.VInputTag(
    cms.InputTag('tauTrigMatchHLT1CaloTau')
    )
allLayer1CaloTaus.genParticleMatch     =  'caloTauMatch'
allLayer1CaloTaus.genJetMatch          =  'caloTauGenJetMatch'
selectedLayer1CaloTaus.src             =  'allLayer1CaloTaus'
selectedLayer1CaloTaus.cut             =  'pt > 10.'
minLayer1CaloTaus.src                  =  'selectedLayer1CaloTaus'
maxLayer1CaloTaus.src                  =  'selectedLayer1CaloTaus'

## standard sequences for caloTau

## production and selection
countLayer1CaloTaus = cms.Sequence(
    minLayer1CaloTaus +
    maxLayer1CaloTaus
) 
tqafLayer1CaloTaus = cms.Sequence(
    allLayer1CaloTaus *
    selectedLayer1CaloTaus *
    countLayer1CaloTaus
)

## standard sequence to produce caloTaus
patCaloTaus = cms.Sequence(
    tqafLayer0Cleaners_withCaloTau *
    tqafLayer0MCTruth_withCaloTau *
    tqafLayer0TrigMatch_withCaloTaus *
    tqafLayer1CaloTaus
)
