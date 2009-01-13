import FWCore.ParameterSet.Config as cms

## import module
from PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cfi import countLayer1Jets

## configure for tqaf
countLayer1Jets.src       = 'selectedLayer1Jets'
countLayer1Jets.maxNumber = 999999
countLayer1Jets.minNumber = 4


