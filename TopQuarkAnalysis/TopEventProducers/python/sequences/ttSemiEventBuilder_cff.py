import FWCore.ParameterSet.Config as cms

#
# produce ttSemiEvent structure with all necessary ingredients
#

## std sequence to produce the ttGenEvent
from TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff import *

## std sequence to produce the ttsemiEventHypotheses
from TopQuarkAnalysis.TopEventProducers.sequences.ttSemiEventHypotheses_cff import *

## configure ttSemiEventBuilder
from TopQuarkAnalysis.TopEventProducers.producers.TtSemiEventBuilder_cfi import *

## make ttSemiEvent structure
makeTtSemiEvent = cms.Sequence(makeTtSemiHyps*ttSemiEvent)

