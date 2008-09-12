import FWCore.ParameterSet.Config as cms

## import module
from PhysicsTools.PatAlgos.selectionLayer1.jetMaxFilter_cfi import maxLayer1Jets

## configure for tqaf
maxLayer1Jets.src       = 'selectedLayer1Jets'
maxLayer1Jets.maxNumber = 999999

## import module
from PhysicsTools.PatAlgos.selectionLayer1.jetMinFilter_cfi import minLayer1Jets

## configure for tqaf
minLayer1Jets.src       = 'selectedLayer1Jets'
minLayer1Jets.minNumber = 2


