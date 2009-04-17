import FWCore.ParameterSet.Config as cms

#
# produce ttSemiLepEvent structure with all necessary ingredients,
# needs ttGenEvent as input
#

## initialize ttGenEvtFilters
from TopQuarkAnalysis.TopEventProducers.sequences.ttGenEventFilters_cff import *

## std sequence to produce the ttSemiLepEventHypotheses
from TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepEvtHypotheses_cff import *

## configure ttSemiLepEventBuilder
from TopQuarkAnalysis.TopEventProducers.producers.TtSemiLepEvtBuilder_cfi import *

## synchronize maxNJets in all hypotheses
ttSemiLepHypGeom              .maxNJets = ttSemiLepEvent.maxNJets
ttSemiLepHypMaxSumPtWMass     .maxNJets = ttSemiLepEvent.maxNJets
ttSemiLepHypWMassMaxSumPt     .maxNJets = ttSemiLepEvent.maxNJets
ttSemiLepJetPartonMatch       .maxNJets = ttSemiLepEvent.maxNJets
findTtSemiLepJetCombMVA       .maxNJets = ttSemiLepEvent.maxNJets
kinFitTtSemiLepEventHypothesis.maxNJets = ttSemiLepEvent.maxNJets

## make ttSemiLepEvent
makeTtSemiLepEvent = cms.Sequence(makeTtSemiLepHypotheses *
                                  ttSemiLepEvent
                                  )

## make ttSemiLepEvent prefiltered for full leptonic decays
makeTtSemiLepEvent_fullLepFilter = cms.Sequence(ttFullyLeptonicFilter *               
                                                makeTtSemiLepHypotheses *
                                                ttSemiLepEvent
                                                )

## make ttSemiLepEvent prefiltered for semi-leptonic decays
makeTtSemiLepEvent_semiLepFilter = cms.Sequence(ttSemiLeptonicFilter *               
                                                makeTtSemiLepHypotheses *
                                                ttSemiLepEvent
                                                )

## make ttSemiLepEvent prefiltered for full hadronic decays
makeTtSemiLepEvent_fullHadFilter = cms.Sequence(ttFullyHadronicFilter *               
                                                makeTtSemiLepHypotheses *
                                                ttSemiLepEvent
                                                )
