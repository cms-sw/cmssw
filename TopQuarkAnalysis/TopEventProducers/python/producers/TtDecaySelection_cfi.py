print "*** Warning: TopQuarkAnalysis/TopEventProducers/python/producers/TtDecaySelection_cfi.py will be deprecated"
print "*** Please use TopQuarkAnalysis/TopSkimming/python/TtDecayChannelFilter_cfi.py instead"
print "*** (you can still use 'genEvent' instead of 'genParticles' as input)"

import TopQuarkAnalysis.TopSkimming.TtDecayChannelFilter_cfi
ttDecaySelection = TopQuarkAnalysis.TopSkimming.TtDecayChannelFilter_cfi.ttDecayChannelFilter.clone()
ttDecaySelection.src = "genEvt"
