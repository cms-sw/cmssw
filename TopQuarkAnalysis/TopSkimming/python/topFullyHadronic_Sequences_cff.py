import FWCore.ParameterSet.Config as cms

sixJetsFilter = cms.EDFilter("EtMinCaloJetCountFilter",
    src = cms.InputTag("iterativeCone5CaloJets"),
    etMin = cms.double(15.0),
    minNumber = cms.uint32(6)
)

twoBJetsFilter = cms.EDFilter("JetTagCountFilter",
    minDiscriminator = cms.double(2.0),
    src = cms.InputTag("trackCountingHighEffJetTags"),
    maxJetEta = cms.double(2.5),
    minNumber = cms.uint32(2),
    minJetEt = cms.double(30.0)
)

topFullyHadronicJets = cms.Sequence(sixJetsFilter)
topFullyHadronicBJets = cms.Sequence(twoBJetsFilter)

