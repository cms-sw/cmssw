import FWCore.ParameterSet.Config as cms

#
# electrons
#

## import module
from PhysicsTools.PatAlgos.selectionLayer1.electronCountFilter_cfi import countLayer1Electrons

## configure for tqaf
countLayer1Electrons.src            = 'selectedLayer1Electrons'
countLayer1Electrons.maxNumber      = 999999
countLayer1Electrons.minNumber      = 0


#
# muons
#

## import module
from PhysicsTools.PatAlgos.selectionLayer1.muonCountFilter_cfi import countLayer1Muons

## configure for tqaf
countLayer1Muons.src                = 'selectedLayer1Muons'
countLayer1Muons.maxNumber          = 999999
countLayer1Muons.minNumber          = 0


#
# taus
#

## import module
from PhysicsTools.PatAlgos.selectionLayer1.tauCountFilter_cfi import countLayer1Taus

## configure for tqaf
countLayer1Taus.src                 = 'selectedLayer1Taus'
countLayer1Taus.maxNumber           = 999999
countLayer1Taus.minNumber           = 0


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




