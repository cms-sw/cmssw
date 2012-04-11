import FWCore.ParameterSet.Config as cms

#
# std filters for specific ttbar decays 
#

import TopQuarkAnalysis.TopSkimming.TtDecayChannelFilter_cfi

## fully-hadronic decay
ttFullHadronicFilter = TopQuarkAnalysis.TopSkimming.TtDecayChannelFilter_cfi.ttDecayChannelFilter.clone()
ttFullHadronicFilter.allowedTopDecays.decayBranchA.electron = False
ttFullHadronicFilter.allowedTopDecays.decayBranchA.muon     = False
ttFullHadronicFilter.allowedTopDecays.decayBranchB.electron = False
ttFullHadronicFilter.allowedTopDecays.decayBranchB.muon     = False

## semi-leptonic decay
ttSemiLeptonicFilter = TopQuarkAnalysis.TopSkimming.TtDecayChannelFilter_cfi.ttDecayChannelFilter.clone()
ttSemiLeptonicFilter.allowedTopDecays.decayBranchA.electron = True
ttSemiLeptonicFilter.allowedTopDecays.decayBranchA.muon     = True
ttSemiLeptonicFilter.allowedTopDecays.decayBranchB.electron = False
ttSemiLeptonicFilter.allowedTopDecays.decayBranchB.muon     = False

## full-leptonic decay
ttFullLeptonicFilter = TopQuarkAnalysis.TopSkimming.TtDecayChannelFilter_cfi.ttDecayChannelFilter.clone()
ttFullLeptonicFilter.allowedTopDecays.decayBranchA.electron = True
ttFullLeptonicFilter.allowedTopDecays.decayBranchA.muon     = True
ttFullLeptonicFilter.allowedTopDecays.decayBranchB.electron = True
ttFullLeptonicFilter.allowedTopDecays.decayBranchB.muon     = True

