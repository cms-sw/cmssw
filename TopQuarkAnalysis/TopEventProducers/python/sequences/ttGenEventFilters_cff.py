print "*** Warning: TopQuarkAnalysis/TopEventProducers/python/sequences/ttGenEventFilters_cff.py will be deprecated"
print "*** Please use TopQuarkAnalysis/TopSkimming/python/ttDecayChannelFilters_cff.py instead"
print "*** (you can still use 'genEvent' instead of 'genParticles' as input)"

from TopQuarkAnalysis.TopSkimming.ttDecayChannelFilter_cff import *
ttFullHadronicFilter.src = "genEvt"
ttSemiLeptonicFilter.src = "genEvt"
ttFullLeptonicFilter.src = "genEvt"
