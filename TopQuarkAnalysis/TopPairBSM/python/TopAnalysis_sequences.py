import FWCore.ParameterSet.Config as cms


## produce ttGenEvent
from TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff import *
## semi-leptonic decay
import TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi
ttMuonicFilter = TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi.ttDecaySelection.clone()
ttMuonicFilter.channel_1  = [0, 1, 0]
ttMuonicFilter.channel_2  = [0, 0, 0]
ttMuonicFilter.tauDecays  = [1, 1, 1]

ttNoMuonicFilter = TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi.ttDecaySelection.clone()
ttNoMuonicFilter.channel_1  = [0, 1, 0]
ttNoMuonicFilter.channel_2  = [0, 0, 0]
ttNoMuonicFilter.tauDecays  = [1, 1, 1]
ttNoMuonicFilter.invert     = cms.bool( True )

## setup HLT filter
from TopQuarkAnalysis.TopPairBSM.BooTopHLTFilter import *

## Analyzer and setup default selection
from TopQuarkAnalysis.TopPairBSM.TopAnalysis_Defaults import *


TopAnalysisMuFilter = cms.Sequence(makeGenEvt+
                           ttMuonicFilter+
                           BooTopHLTFilter+
                           TopAnalyzer)

TopAnalysisNoMuFilter = cms.Sequence(makeGenEvt+
                           ttNoMuonicFilter+
                           BooTopHLTFilter+
                           TopAnalyzer)

TopAnalysis = cms.Sequence(makeGenEvt+
                           BooTopHLTFilter+
                           TopAnalyzer)
