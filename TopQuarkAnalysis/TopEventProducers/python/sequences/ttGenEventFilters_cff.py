import FWCore.ParameterSet.Config as cms

#
# std filteres for specific ttbar decays 
#

## full hadronic decay
import TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi
ttFullyHadronicFilter = TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi.ttDecaySelection.clone()
ttFullyHadronicFilter.channel_1 = [0, 0, 0]
ttFullyHadronicFilter.channel_2 = [0, 0, 0]
ttFullyHadronicFilter.tauDecays = [0, 0, 0]

## semi-leptonic decay
import TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi
ttSemiLeptonicFilter = TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi.ttDecaySelection.clone()
ttSemiLeptonicFilter.channel_1  = [1, 1, 0]
ttSemiLeptonicFilter.channel_2  = [0, 0, 0]
ttSemiLeptonicFilter.tauDecays  = [0, 0, 0]

## full leptonic decay
import TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi
ttFullyLeptonicFilter = TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi.ttDecaySelection.clone()
ttFullyLeptonicFilter.channel_1 = [1, 1, 0]
ttFullyLeptonicFilter.channel_2 = [1, 1, 0]
ttFullyLeptonicFilter.tauDecays = [1, 1, 0]

