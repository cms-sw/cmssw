import FWCore.ParameterSet.Config as cms


## produce ttGenEvent
from TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff import *
## semi-leptonic decay
import TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi

ttMuonicFilter = TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi.ttDecaySelection.clone()
ttMuonicFilter.allowedTopDecays.decayBranchA.electron = cms.bool(False)
ttMuonicFilter.allowedTopDecays.decayBranchA.muon     = cms.bool(True)
ttMuonicFilter.allowedTopDecays.decayBranchA.tau      = cms.bool(False)
ttMuonicFilter.allowedTopDecays.decayBranchB.electron = cms.bool(False)
ttMuonicFilter.allowedTopDecays.decayBranchB.muon     = cms.bool(False)
ttMuonicFilter.allowedTopDecays.decayBranchB.tau      = cms.bool(False)
ttMuonicFilter.restrictTauDecays.leptonic	      = cms.bool(True)
ttMuonicFilter.restrictTauDecays.oneProng             = cms.bool(True)
ttMuonicFilter.restrictTauDecays.threeProng           = cms.bool(True)

## invert selection
ttNoMuonicFilter = TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi.ttDecaySelection.clone()
ttNoMuonicFilter.allowedTopDecays.decayBranchA.electron = cms.bool(False)
ttNoMuonicFilter.allowedTopDecays.decayBranchA.muon     = cms.bool(True)
ttNoMuonicFilter.allowedTopDecays.decayBranchA.tau      = cms.bool(False)
ttNoMuonicFilter.allowedTopDecays.decayBranchB.electron = cms.bool(False)
ttNoMuonicFilter.allowedTopDecays.decayBranchB.muon     = cms.bool(False)
ttNoMuonicFilter.allowedTopDecays.decayBranchB.tau      = cms.bool(False)
ttNoMuonicFilter.restrictTauDecays.leptonic             = cms.bool(True)
ttNoMuonicFilter.restrictTauDecays.oneProng             = cms.bool(True)
ttNoMuonicFilter.restrictTauDecays.threeProng           = cms.bool(True)
ttNoMuonicFilter.invert                                 = cms.bool(True)


## setup HLT filter
from TopQuarkAnalysis.TopPairBSM.BooTopHLTFilter import *

## Analyzer and setup default selection
from TopQuarkAnalysis.TopPairBSM.TopAnalysis_Defaults import *
from TopQuarkAnalysis.TopPairBSM.ABCDAnalysis_Defaults import *
from TopQuarkAnalysis.TopPairBSM.HighAnalysis_Defaults import *


TopAnalysisMuFilter = cms.Sequence(
                           ttMuonicFilter+
                           BooTopHLTFilter+
                           TopAnalyzer)

TopAnalysisNoMuFilter = cms.Sequence(
                           ttNoMuonicFilter+
                           BooTopHLTFilter+
                           TopAnalyzer)

TopAnalysis = cms.Sequence(BooTopHLTFilter+
			   TopAnalyzer)

ABCDAnalysis = cms.Sequence(ABCDAnalyzer)

HighMAnalysis = cms.Sequence(HighMAnalyzer)
