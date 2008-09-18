import FWCore.ParameterSet.Config as cms

#---------------------------------------
# object cleaning
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.caloTauCleaner_cfi import allLayer0CaloTaus

## standard sequence for caloTauCleaner
tqafLayer0Cleaners_withCaloTau = cms.Sequence(
    allLayer0CaloTaus
)

#---------------------------------------
# mc matching
#---------------------------------------
import PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi

## clone module(s)
caloTauMatch = PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi.tauMatch.clone()
caloTauMatch.src = 'allLayer0CaloTaus'

## standard sequence for mcTruthMatching
tqafLayer0MCTruth_withCaloTau = cms.Sequence(
    caloTauMatch
)

#---------------------------------------
# trigger matching
#---------------------------------------
from TopQuarkAnalysis.TopObjectProducers.full.tqafLayer0_triggerMatching_cfi import *

## standard sequence for triggerMatching
tqafLayer0TrigMatch_withCaloTaus = cms.Sequence(
    patHLT1Tau * tauTrigMatchHLT1CaloTau +
    patHLT2TauPixel * tauTrigMatchHLT2CaloTauPixel
)

#---------------------------------------
# caloTau producer and selectors
#---------------------------------------
import PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi
import PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi
import PhysicsTools.PatAlgos.selectionLayer1.tauMinFilter_cfi
import PhysicsTools.PatAlgos.selectionLayer1.tauMaxFilter_cfi

## clone module(s)
allLayer1CaloTaus      = PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi.allLayer1Taus.clone()
selectedLayer1CaloTaus = PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi.selectedLayer1Taus.clone()
minLayer1CaloTaus      = PhysicsTools.PatAlgos.selectionLayer1.tauMinFilter_cfi.minLayer1Taus.clone()
maxLayer1CaloTaus      = PhysicsTools.PatAlgos.selectionLayer1.tauMaxFilter_cfi.maxLayer1Taus.clone()

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
tqafLayer1_caloTaus = cms.Sequence(
    tqafLayer0Cleaners_withCaloTau *
    tqafLayer0MCTruth_withCaloTau *
    tqafLayer0TrigMatch_withCaloTaus *
    tqafLayer1CaloTaus
)
