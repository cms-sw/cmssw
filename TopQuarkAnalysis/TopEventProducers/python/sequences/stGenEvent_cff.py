import FWCore.ParameterSet.Config as cms

#
# produce stGenEvent with all necessary ingredients
#
from TopQuarkAnalysis.TopEventProducers.producers.TopInitSubset_cfi import *
from TopQuarkAnalysis.TopEventProducers.producers.TopDecaySubset_cfi import *
from TopQuarkAnalysis.TopEventProducers.producers.StGenEvtProducer_cfi import *

makeGenEvt = cms.Sequence(initSubset*decaySubset*genEvtSingleTop)

