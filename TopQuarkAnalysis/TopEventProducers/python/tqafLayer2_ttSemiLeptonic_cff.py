import FWCore.ParameterSet.Config as cms

#
# tqaf Layer 2 for semi-leptonic event selections
#
# produce ttGenEvt and initialize ttGenEvtFilters
from TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff import *
from TopQuarkAnalysis.TopEventProducers.sequences.ttGenEventFilters_cff import *
# apply objects count selection
from TopQuarkAnalysis.TopEventProducers.sequences.numJetFilter_ttSemiLeptonic_cff import *
from TopQuarkAnalysis.TopEventProducers.sequences.numLeptonFilter_ttSemiLeptonic_cff import *
# produce event hypotheses
from TopQuarkAnalysis.TopEventProducers.producers.TtSemiEvtSolProducer_cfi import *
from TopQuarkAnalysis.TopEventProducers.sequences.tqafEventHyp_ttSemiLeptonic_cff import *
#
tqafLayer2_ttSemiLeptonic = cms.Sequence(makeGenEvt*makeTtSemiEvent*solutions)
tqafLayer2_ttSemiLeptonic_SemiLepFilter = cms.Sequence(makeGenEvt*makeTtSemiEvent*ttSemiLeptonicFilter*solutions)
tqafLayer2_ttSemiLeptonic_FullLepFilter = cms.Sequence(makeGenEvt*ttFullyLeptonicFilter*solutions)
tqafLayer2_ttSemiLeptonic_FullHadFilter = cms.Sequence(makeGenEvt*ttFullyHadronicFilter*solutions)

