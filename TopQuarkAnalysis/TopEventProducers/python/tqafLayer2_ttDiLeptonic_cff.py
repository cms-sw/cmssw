import FWCore.ParameterSet.Config as cms

#
# tqaf Layer 2 for semi-leptonic event selections
#

## produce ttGenEvt and initialize ttGenEvtFilters
from TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff import *
from TopQuarkAnalysis.TopEventProducers.sequences.ttGenEventFilters_cff import *

## apply objects count selection
from TopQuarkAnalysis.TopEventProducers.sequences.numJetFilter_ttFullLeptonic_cff import *
from TopQuarkAnalysis.TopEventProducers.sequences.numLepFilter_ttFullLeptonic_cff import *

## produce event hypotheses
from TopQuarkAnalysis.TopEventProducers.producers.TtDilepEvtSolProducer_cfi import *

## make tqaf layer2
tqafLayer2_ttDiLeptonic = cms.Sequence(makeGenEvt*solutions)

## make tqaf layer2 filtered for full leptonic decays
tqafLayer2_ttDileptonic_FullLepFilter = cms.Sequence(makeGenEvt*ttFullyLeptonicFilter*solutions)

## make tqaf layer2 filtered for semi-leptonic decays
tqafLayer2_ttDileptonic_SemiLepFilter = cms.Sequence(makeGenEvt*ttSemiLeptonicFilter *solutions)

## make tqaf layer2 filtered for full hadronic decays
tqafLayer2_ttDileptonic_FullHadFilter = cms.Sequence(makeGenEvt*ttFullyHadronicFilter*solutions)

