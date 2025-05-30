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

#### change the used triggers for run2 ####
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
    src = cms.InputTag("slimmedMuons"),
    cut = cms.string("pt > 8 && abs(eta) < 2.5"),
    filter = cms.bool(True)
)


# For impact parameter (w.r.t. to PV) requirements, a vector collection is needed, therefore only dB < 0.2 required.
# The default requirements (in C++):
# 1) fabs(recoMu.muonBestTrack()->dxy(vertex->position())) < 0.2 ----> similar to dB < 0.2
# 2) fabs(recoMu.muonBestTrack()->dz(vertex->position())) < 0.5
patMuonsAfterTightID = cms.EDFilter("PATMuonSelector",
    src = cms.InputTag("patMuonsAfterKinCuts"),
    cut = cms.string(
    "isPFMuon && isGlobalMuon"
    " && muonID('GlobalMuonPromptTight')"
    " && numberOfMatchedStations > 1"
    " && innerTrack.hitPattern.trackerLayersWithMeasurement > 5"
    " && innerTrack.hitPattern.numberOfValidPixelHits > 0"
    " && dB < 0.2"
    ),
    filter = cms.bool(True)
)

patMuonsAfterMediumID = cms.EDFilter("PATMuonSelector",
    src = cms.InputTag("patMuonsAfterKinCuts"),
    cut = cms.string("isMediumMuon"),
    filter = cms.bool(True)
)

patMuonsAfterLooseID = cms.EDFilter("PATMuonSelector",
    src = cms.InputTag("patMuonsAfterKinCuts"),
    cut = cms.string("isLooseMuon"),
    filter = cms.bool(True)
)

patMuonsAfterID = patMuonsAfterLooseID.clone()

ZmumuCandidates = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    # require one of the muons with pT > 17 GeV, and an invariant mass > 20 GeV
    cut = cms.string('charge = 0 & max(daughter(0).pt, daughter(1).pt) > 17 & mass > 20 & daughter(0).isGlobalMuon & daughter(1).isGlobalMuon'),
    decay = cms.string("patMuonsAfterID@+ patMuonsAfterID@-")
)


ZmumuCandidatesFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("ZmumuCandidates"),
    minNumber = cms.uint32(1)
    # filter = cms.bool(True)
)

selectedMuonsForEmbedding = cms.EDProducer("MuMuForEmbeddingSelector",
    ZmumuCandidatesCollection = cms.InputTag("ZmumuCandidates"),
    use_zmass = cms.bool(False),
    inputTagVertex = cms.InputTag("offlinePrimaryVertices"),
    inputTagBeamSpot = cms.InputTag("offlineBeamSpot"),
    PuppiMet = cms.InputTag("slimmedMETsPuppi"),
    Met = cms.InputTag("slimmedMETs"),
)

makePatMuonsZmumuSelection = cms.Sequence(
    doubleMuonHLTTrigger
    + patMuonsAfterKinCuts
    + patMuonsAfterID
    + ZmumuCandidates
    + ZmumuCandidatesFilter
    + selectedMuonsForEmbedding
)
