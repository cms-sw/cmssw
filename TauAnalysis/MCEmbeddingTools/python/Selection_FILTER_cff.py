"""
This config fragment filters for a muon pair which can be used for tau embedding.
It selects muons based on trigger conditions, kinematic cuts, and identification criteria.
It then produces a collection of muons suitable for embedding (selectedMuonsForEmbedding), including their kinematic properties.
This is then later used in the LHE step to simulate two taus with slightly adjusted kinematics.
To use this config fragment, a cmsDriver command like the following can be used:
```
cmsDriver.py \
    --step RAW2DIGI,L1Reco,RECO,PAT,FILTER:TauAnalysis/MCEmbeddingTools/Selection_FILTER_cff.makePatMuonsZmumuSelection \
    --processName SELECT \
    --data \
    --scenario pp \
    --eventcontent TauEmbeddingSelection \
    --datatier RAWRECO \
    --era ... \
    --conditions ... \
    --filein ... \
    --fileout ...
```
"""

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run2_HLTconditions_2016_cff import (
    run2_HLTconditions_2016,
)
from Configuration.Eras.Modifier_run2_HLTconditions_2017_cff import (
    run2_HLTconditions_2017,
)
from Configuration.Eras.Modifier_run2_HLTconditions_2018_cff import (
    run2_HLTconditions_2018,
)
from Configuration.StandardSequences.PAT_cff import *
from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import patMuons

## Trigger requirements
doubleMuonHLTTrigger = cms.EDFilter("TriggerResultsFilter",
    hltResults = cms.InputTag("TriggerResults","","HLT"),
    l1tResults = cms.InputTag(""),
    throw = cms.bool(False),
    triggerConditions = cms.vstring("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8_v*") # unprescaled trigger for 2018,22,23,24 (see https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLT2018, https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLT2022, https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLT2023, https://twiki.cern.ch/twiki/bin/view/CMS/MuonHLT2024)
)

# change the used triggers for run2
# Use two different triggers as the Mass8 one has a higer luminosity in 2017 according to https://cmshltinfo.app.cern.ch/summary?search=HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass&year=2017&paths=true&prescaled=true&stream-types=Physics
# probably because he was already active in earlier runs than the Mass3p8 trigger
# Both are unprescaled
run2_HLTconditions_2017.toModify(doubleMuonHLTTrigger,
                                 triggerConditions = ["HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v* OR HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_v*"])

# Both are unprescaled according to https://cmshltinfo.app.cern.ch/summary?search=HLT_Mu17_TrkIsoVVL_&year=2016&paths=true&prescaled=true&stream-types=Physics
run2_HLTconditions_2016.toModify(doubleMuonHLTTrigger,
                                 triggerConditions = ["HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v* OR HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v*"])

## Muon selection
patMuonsAfterKinCuts = cms.EDFilter("PATMuonSelector",
    src = cms.InputTag("slimmedMuons", "", "SELECT"),
    cut = cms.string("pt > 8 && abs(eta) < 2.5"),
    filter = cms.bool(True)
)

## require loose muon ID
patMuonsAfterID = cms.EDFilter("PATMuonSelector",
    src = cms.InputTag("patMuonsAfterKinCuts"),
    cut = cms.string("isLooseMuon"),
    filter = cms.bool(True)
)

## create Z->mumu candidates
ZmumuCandidates = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    # require one of the muons with pT > 17 GeV, and an invariant mass > 20 GeV
    cut = cms.string('charge = 0 & max(daughter(0).pt, daughter(1).pt) > 17 & mass > 20 & daughter(0).isGlobalMuon & daughter(1).isGlobalMuon'),
    decay = cms.string("patMuonsAfterID@+ patMuonsAfterID@-")
)

## require at least one Z->mumu candidate
ZmumuCandidatesFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("ZmumuCandidates"),
    minNumber = cms.uint32(1)
    # filter = cms.bool(True)
)

## Create a collection of muons suitable for embedding
# The MuMuForEmbeddingSelector therefore selects the muons from the Z->mumu candidates,
# where the Z->mumu candidate mass is the highest.
selectedMuonsForEmbedding = cms.EDProducer("MuMuForEmbeddingSelector",
    ZmumuCandidatesCollection = cms.InputTag("ZmumuCandidates"),
    inputTagVertex = cms.InputTag("offlinePrimaryVertices"),
    inputTagBeamSpot = cms.InputTag("offlineBeamSpot"),
    PuppiMet = cms.InputTag("slimmedMETsPuppi", "", "SELECT"),
    Met = cms.InputTag("slimmedMETs", "", "SELECT"),
)

makePatMuonsZmumuSelection = cms.Sequence(
    doubleMuonHLTTrigger
    + patMuonsAfterKinCuts
    + patMuonsAfterID
    + ZmumuCandidates
    + ZmumuCandidatesFilter
    + selectedMuonsForEmbedding
)
