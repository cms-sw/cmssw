import FWCore.ParameterSet.Config as cms

#
# produce ttGenEvent with all necessary ingredients
#
from TopQuarkAnalysis.TopEventProducers.producers.TopInitSubset_cfi import *
from TopQuarkAnalysis.TopEventProducers.producers.TopDecaySubset_cfi import *
from TopQuarkAnalysis.TopEventProducers.producers.TtGenEvtProducer_cfi import *

makeGenEvt = cms.Sequence(initSubset*decaySubset*genEvt)

