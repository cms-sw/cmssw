import FWCore.ParameterSet.Config as cms

#
# produce ttSemiEvent structure with all necessary ingredients
#

## std sequence to produce the ttGenEvent
from TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff import *

## initialize ttGenEvtFilters
from TopQuarkAnalysis.TopEventProducers.sequences.ttGenEventFilters_cff import *

## std sequence to produce the ttsemiEventHypotheses
from TopQuarkAnalysis.TopEventProducers.sequences.ttSemiEventHypotheses_cff import *

## configure ttSemiEventBuilder
from TopQuarkAnalysis.TopEventProducers.producers.TtSemiLepEvtBuilder_cfi import *

## make ttSemiEvent
makeTtSemiEvent = cms.Sequence(makeGenEvt *
                               makeTtSemiLepHypotheses *
                               ttSemiLepEvent
                               )

## make ttSemiEvent prefiltered for full leptonic decays
makeTtSemiEvent_fullLepFilter = cms.Sequence(makeGenEvt *
                               ttFullyLeptonicFilter *               
                               makeTtSemiLepHypotheses *
                               ttSemiLepEvent
                               )

## make ttSemiEvent prefiltered for semi-leptonic decays
makeTtSemiEvent_semiLepFilter = cms.Sequence(makeGenEvt *
                               ttSemiLeptonicFilter *               
                               makeTtSemiLepHypotheses *
                               ttSemiLepEvent
                               )

## make ttSemiEvent prefiltered for full hadronic decays
makeTtSemiEvent_fullHadFilter = cms.Sequence(makeGenEvt *
                               ttFullyHadronicFilter *               
                               makeTtSemiLepHypotheses *
                               ttSemiLepEvent
                               )
