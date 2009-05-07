import FWCore.ParameterSet.Config as cms

#
# produce ttSemiLepEvent structure with all necessary ingredients,
# needs ttGenEvent as input
#

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
