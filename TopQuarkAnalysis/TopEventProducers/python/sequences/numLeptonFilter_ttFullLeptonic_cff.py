import FWCore.ParameterSet.Config as cms

#
# electrons
#

## import module
from PhysicsTools.PatAlgos.selectionLayer1.electronMaxFilter_cfi import maxLayer1Electrons

## configure for tqaf
maxLayer1Electrons.src            = 'selectedLayer1Electrons'
maxLayer1Electrons.maxNumber      = 999999

## import module
from PhysicsTools.PatAlgos.selectionLayer1.electronMinFilter_cfi import minLayer1Electrons

## configure for tqaf
minLayer1Electrons.src            = 'selectedLayer1Electrons'
minLayer1Electrons.minNumber      = 0


#
# muons
#

## import module
from PhysicsTools.PatAlgos.selectionLayer1.muonMaxFilter_cfi import maxLayer1Muons

## configure for tqaf
maxLayer1Muons.src                = 'selectedLayer1Muons'
maxLayer1Muons.maxNumber          = 999999

## import module
from PhysicsTools.PatAlgos.selectionLayer1.muonMinFilter_cfi import minLayer1Muons

## configure for tqaf
minLayer1Muons.src                = 'selectedLayer1Muons'
minLayer1Muons.minNumber          = 0


#
# taus
#

## import module
from PhysicsTools.PatAlgos.selectionLayer1.tauMaxFilter_cfi import maxLayer1Taus

## configure for tqaf
maxLayer1Taus.src                 = 'selectedLayer1Taus'
maxLayer1Taus.maxNumber           = 999999

## import module
from PhysicsTools.PatAlgos.selectionLayer1.tauMinFilter_cfi import minLayer1Taus

## configure for tqaf
minLayer1Taus.src                 = 'selectedLayer1Taus'
minLayer1Taus.minNumber           = 0


#
# leptons in general
#

## import module
from PhysicsTools.PatAlgos.selectionLayer1.leptonCountFilter_cfi import countLayer1Leptons

## configure for tqaf
countLayer1Leptons.electronSource = 'selectedLayer1Electrons'
countLayer1Leptons.muonSource     = 'selectedLayer1Muons'
countLayer1Leptons.tauSource      = 'selectedLayer1Taus'
countLayer1Leptons.countElectrons = True
countLayer1Leptons.countMuons     = True
countLayer1Leptons.countTaus      = False
countLayer1Leptons.minNumber      = 2
countLayer1Leptons.maxNumber      = 999999




