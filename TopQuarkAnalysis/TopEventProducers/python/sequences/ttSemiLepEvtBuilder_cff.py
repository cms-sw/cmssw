import FWCore.ParameterSet.Config as cms

#
# produce ttSemiEvent structure with all necessary ingredients,
# needs ttGenEvent as input
#

## initialize ttGenEvtFilters
from TopQuarkAnalysis.TopEventProducers.sequences.ttGenEventFilters_cff import *

## std sequence to produce the ttsemiEventHypotheses
from TopQuarkAnalysis.TopEventProducers.sequences.ttSemiEventHypotheses_cff import *

## configure ttSemiEventBuilder
from TopQuarkAnalysis.TopEventProducers.producers.TtSemiLepEvtBuilder_cfi import *

## make ttSemiEvent
makeTtSemiLepEvent = cms.Sequence(makeTtSemiLepHypotheses *
                                  ttSemiLepEvent
                                  )

## make ttSemiEvent prefiltered for full leptonic decays
makeTtSemiLepEvent_fullLepFilter = cms.Sequence(ttFullyLeptonicFilter *               
                                                makeTtSemiLepHypotheses *
                                                ttSemiLepEvent
                                                )

## make ttSemiEvent prefiltered for semi-leptonic decays
makeTtSemiLepEvent_semiLepFilter = cms.Sequence(ttSemiLeptonicFilter *               
                                                makeTtSemiLepHypotheses *
                                                ttSemiLepEvent
                                                )

## make ttSemiEvent prefiltered for full hadronic decays
makeTtSemiLepEvent_fullHadFilter = cms.Sequence(ttFullyHadronicFilter *               
                                                makeTtSemiLepHypotheses *
                                                ttSemiLepEvent
                                                )
