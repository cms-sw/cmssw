import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.PAT_cff import *

from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import patMuons
from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *


## Trigger requirements
doubleMuonHLTTrigger = cms.EDFilter("TriggerResultsFilter",
    hltResults = cms.InputTag("TriggerResults","","HLT"),
    l1tResults = cms.InputTag(""),
    throw = cms.bool(False),
    triggerConditions = cms.vstring("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v* OR HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8_v*") # from 2017 on (up to Run 3, it seems)
    # triggerConditions = cms.vstring("HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_v* OR HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ_v*") # for 2016
)



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
