import FWCore.ParameterSet.Config as cms

# WATCH OUT: no isolation!
#module trackFilter = PtMinTrackCountFilter {
#  InputTag src = ctfWithMaterialTracks
#  uint32 minNumber = 1
#  double ptMin = 20.
#}
muonFilter = cms.EDFilter("PtMinMuonCountFilter",
    src = cms.InputTag("muons"),
    ptMin = cms.double(20.0),
    minNumber = cms.uint32(1)
)

singleJetFilterMuon = cms.EDFilter("EtMinCaloJetCountFilter",
    src = cms.InputTag("iterativeCone5CaloJets"),
    etMin = cms.double(30.0),
    minNumber = cms.uint32(1)
)

doubleJetFilterMuon = cms.EDFilter("EtMinCaloJetCountFilter",
    src = cms.InputTag("iterativeCone5CaloJets"),
    etMin = cms.double(15.0),
    minNumber = cms.uint32(2)
)

topSemiLepMuonPlus1Jet = cms.Sequence(singleJetFilterMuon+muonFilter)
topSemiLepMuonPlus2Jets = cms.Sequence(doubleJetFilterMuon+muonFilter)

