print "*** Warning: TopQuarkAnalysis/TopSkimming/python/TtFullyLeptonicFilter_cfi.py will be deprecated"
print "*** Please use TopQuarkAnalysis/TopSkimming/python/TtDecayChannelFilter_cfi.py or"
print "*** TopQuarkAnalysis/TopSkimming/python/ttDecayChannelFilter_cff.py instead"

from TopQuarkAnalysis.TopSkimming.ttDecayChannelFilters_cff import ttFullLeptonicFilter
ttFullyLeptonicFilter = ttFullLeptonicFilter
