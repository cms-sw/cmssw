import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# tqaf Layer 2 common
#-------------------------------------------------

## produce ttGenEvent
from TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff import *

## make tqaf layer2 common
tqafLayer2_common = cms.Sequence(makeGenEvt)
