import FWCore.ParameterSet.Config as cms

#
# L0 input
#

## import module
from PhysicsTools.PatAlgos.cleaningLayer0.caloMetCleaner_cfi import allLayer0METs

## configure for tqaf
allLayer0METs.metSource = 'corMetType1Icone5'
