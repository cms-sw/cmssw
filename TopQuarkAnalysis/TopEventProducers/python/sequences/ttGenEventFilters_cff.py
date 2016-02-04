import FWCore.ParameterSet.Config as cms

#
# std filteres for specific ttbar decays 
#

## full hadronic decay
import TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi

ttFullHadronicFilter = TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi.ttDecaySelection.clone()
ttFullHadronicFilter.allowedTopDecays.decayBranchA.electron = False
ttFullHadronicFilter.allowedTopDecays.decayBranchA.muon     = False
ttFullHadronicFilter.allowedTopDecays.decayBranchB.electron = False
ttFullHadronicFilter.allowedTopDecays.decayBranchB.muon     = False

## semi-leptonic decay
ttSemiLeptonicFilter = TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi.ttDecaySelection.clone()
ttSemiLeptonicFilter.allowedTopDecays.decayBranchA.electron = True
ttSemiLeptonicFilter.allowedTopDecays.decayBranchA.muon     = True
ttSemiLeptonicFilter.allowedTopDecays.decayBranchB.electron = False
ttSemiLeptonicFilter.allowedTopDecays.decayBranchB.muon     = False

## full leptonic decay
ttFullLeptonicFilter = TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi.ttDecaySelection.clone()
ttFullLeptonicFilter.allowedTopDecays.decayBranchA.electron = True
ttFullLeptonicFilter.allowedTopDecays.decayBranchA.muon     = True
ttFullLeptonicFilter.allowedTopDecays.decayBranchB.electron = True
ttFullLeptonicFilter.allowedTopDecays.decayBranchB.muon     = True
