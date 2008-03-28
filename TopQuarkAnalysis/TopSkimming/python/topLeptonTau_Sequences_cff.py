import FWCore.ParameterSet.Config as cms

eTauFilter = cms.EDFilter("TopLeptonTauFilter",
    Tausrc = cms.InputTag("coneIsolationTauJetTags"),
    Elecsrc = cms.InputTag("pixelMatchGsfElectrons"),
    MuonPtmin = cms.double(15.0),
    CaloJetsrc = cms.InputTag("iterativeCone5CaloJets"),
    NminMuon = cms.int32(0),
    NminElec = cms.int32(0),
    ElecPtmin = cms.double(15.0),
    NminCaloJet = cms.int32(2),
    Muonsrc = cms.InputTag("muons"),
    ElecFilter = cms.bool(True),
    TauFilter = cms.bool(True),
    MuonFilter = cms.bool(False),
    TauLeadTkPtmin = cms.double(15.0),
    CaloJetPtmin = cms.double(15.0),
    JetFilter = cms.bool(True),
    NminTau = cms.int32(1)
)

muTauFilter = cms.EDFilter("TopLeptonTauFilter",
    Tausrc = cms.InputTag("coneIsolationTauJetTags"),
    Elecsrc = cms.InputTag("pixelMatchGsfElectrons"),
    MuonPtmin = cms.double(15.0),
    CaloJetsrc = cms.InputTag("iterativeCone5CaloJets"),
    NminMuon = cms.int32(0),
    NminElec = cms.int32(0),
    ElecPtmin = cms.double(15.0),
    NminCaloJet = cms.int32(2),
    Muonsrc = cms.InputTag("muons"),
    ElecFilter = cms.bool(False),
    TauFilter = cms.bool(True),
    MuonFilter = cms.bool(True),
    TauLeadTkPtmin = cms.double(15.0),
    CaloJetPtmin = cms.double(15.0),
    JetFilter = cms.bool(True),
    NminTau = cms.int32(1)
)

topLeptonTauMuTau = cms.Sequence(muTauFilter)
topLeptonTauETau = cms.Sequence(eTauFilter)

