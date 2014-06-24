import FWCore.ParameterSet.Config as cms

l1MuonRecoTreeProducer = cms.EDAnalyzer("L1MuonRecoTreeProducer",
  maxMuon = cms.uint32(20),
  muonTag = cms.InputTag("muons"),

  maxRcpHit = cms.uint32(100),
  rpcHitTag = cms.InputTag("rpcRecHits"),

  #---------------------------------------------------------------------
  # TRIGGER MATCHING CONFIGURATION
  #---------------------------------------------------------------------
  # flag to turn trigger matching on / off
  triggerMatching = cms.untracked.bool(False),
  # maximum delta R between trigger object and muon
  triggerMaxDeltaR = cms.double(.1),
  # trigger to match to, may use regexp wildcard as supported by ROOT's 
  # TString; up to now the first found match (per run) is used.
  isoTriggerNames = cms.vstring(
        "HLT_IsoMu24_eta2p1_v*",
        "HLT_IsoMu24_v*"
        ),
  triggerNames = cms.vstring(
        "HLT_Mu30_v*",
        "HLT_Mu40_v*"
        ),

  # data best guess: change for MC!
  triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD", "", "HLT"),
  # name of the hlt process (same as above):
  triggerProcessLabel = cms.untracked.string("HLT"),
)

