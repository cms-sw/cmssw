import FWCore.ParameterSet.Config as cms

# WATCH OUT: no isolation!
#module trackFilter = PtMinTrackCountFilter {
#  InputTag src = ctfWithMaterialTracks
#  uint32 minNumber = 1
#  double ptMin = 20.
#}
electronFilter = cms.EDFilter("PtMinGsfElectronCountFilter",
    src = cms.InputTag("pixelMatchGsfElectrons"),
    ptMin = cms.double(20.0),
    minNumber = cms.uint32(1)
)

singleJetFilterElectron = cms.EDFilter("EtMinCaloJetCountFilter",
    src = cms.InputTag("iterativeCone5CaloJets"),
    etMin = cms.double(30.0),
    minNumber = cms.uint32(1)
)

doubleJetFilterElectron = cms.EDFilter("EtMinCaloJetCountFilter",
    src = cms.InputTag("iterativeCone5CaloJets"),
    etMin = cms.double(15.0),
    minNumber = cms.uint32(2)
)

topSemiLepElectronPlus1Jet = cms.Sequence(singleJetFilterElectron+electronFilter)
topSemiLepElectronPlus2Jets = cms.Sequence(doubleJetFilterElectron+electronFilter)

