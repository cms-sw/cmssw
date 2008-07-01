import FWCore.ParameterSet.Config as cms

import TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi
ttFullyHadronicFilter = TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi.ttDecaySelection.clone()
import TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi
ttSemiLeptonicFilter = TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi.ttDecaySelection.clone()
import TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi
ttFullyLeptonicFilter = TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi.ttDecaySelection.clone()
ttFullyHadronicFilter.channel_1 = [0, 0, 0]
ttFullyHadronicFilter.channel_2 = [0, 0, 0]
ttFullyHadronicFilter.tauDecays = [0, 0, 0]
ttSemiLeptonicFilter.channel_1  = [1, 1, 0]
ttSemiLeptonicFilter.channel_2  = [0, 0, 0]
ttSemiLeptonicFilter.tauDecays  = [0, 0, 0]
ttFullyLeptonicFilter.channel_1 = [1, 1, 0]
ttFullyLeptonicFilter.channel_2 = [1, 1, 0]
ttFullyLeptonicFilter.tauDecays = [1, 1, 0]

