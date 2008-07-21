import FWCore.ParameterSet.Config as cms

#---------------------------------------
# Electron
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import allLayer1Electrons

## input source
allLayer1Electrons.electronSource = cms.InputTag("allLayer0Electrons")
## embed AOD objects?
allLayer1Electrons.embedTrack = cms.bool(False)
allLayer1Electrons.embedGsfTrack = cms.bool(False)
allLayer1Electrons.embedSuperCluster = cms.bool(False)
## mc matching
allLayer1Electrons.addGenMatch = cms.bool(True)
allLayer1Electrons.embedGenMatchd = cms.bool(False)
allLayer1Electrons.genParticleMatch = cms.InputTag("electronMatch")
## resolution
allLayer1Electrons.addResolutions = cms.bool(True)
allLayer1Electrons.useNNResolutions = cms.bool(False)
allLayer1Electrons.electronResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_electron.root')
## isolation for tracker
allLayer1Electrons.isolation.tracker = cms.PSet(
    src    = cms.InputTag("layer0ElectronIsolations", "eleIsoDepositTk"),
    deltaR = cms.double(0.3)
    )
## isolation for ecal
allLayer1Electrons.isolation.ecal = cms.PSet(
    src    = cms.InputTag("layer0ElectronIsolations", "eleIsoDepositEcalFromHits"),
    deltaR = cms.double(0.4)
    )
## isolation for hcal
allLayer1Electrons.isolation.hcal = cms.PSet(
    src    = cms.InputTag("layer0ElectronIsolations", "eleIsoDepositHcalFromHits"),
    deltaR = cms.double(0.4)
    )
##store deposits
allLayer1Electrons.isoDeposits = cms.PSet(
    tracker= cms.InputTag("layer0ElectronIsolations", "eleIsoDepositTk"),
    ecal   = cms.InputTag("layer0ElectronIsolations", "eleIsoDepositEcalFromHits"),
    hcal   = cms.InputTag("layer0ElectronIsolations", "eleIsoDepositHcalFromHits")
    )


#---------------------------------------
# Muon
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import allLayer1Muons

## input source
allLayer1Muons.muonSource = cms.InputTag("allLayer0Muons")
## embed AOD objects?
allLayer1Muons.embedTrack = cms.bool(False)
allLayer1Muons.embedCombinedMuon = cms.bool(False)
allLayer1Muons.embedStandAloneMuon = cms.bool(False)
## mc matching
allLayer1Muons.addGenMatch = cms.bool(True)
allLayer1Muons.embedGenMatch = cms.bool(False)
allLayer1Muons.genParticleMatch = cms.InputTag("muonMatch")
## resolution
allLayer1Muons.addResolutions = cms.bool(True)
allLayer1Muons.useNNResolutions = cms.bool(False)
allLayer1Muons.muonResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_muon.root')
## isolation for tracker
allLayer1Muons.isolation.tracker = cms.PSet(
    src    = cms.InputTag("layer0MuonIsolations", "muIsoDepositTk"),
    deltaR = cms.double(0.3)
    )
## isolation for ecal
allLayer1Muons.isolation.ecal = cms.PSet(
    src    = cms.InputTag("layer0MuonIsolations", "muIsoDepositCalByAssociatorTowersecal"),
    deltaR = cms.double(0.3)
    )
## isolation for hcal
allLayer1Muons.isolation.hcal = cms.PSet(
    src    = cms.InputTag("layer0MuonIsolations", "muIsoDepositCalByAssociatorTowershcal"),
    deltaR = cms.double(0.3)
    )
## isolation for ho
allLayer1Muons.isolation.user = cms.VPSet(
    cms.PSet(
    src    = cms.InputTag("layer0MuonIsolations", "muIsoDepositCalByAssociatorTowersho"),
    deltaR = cms.double(0.3)
    ),
    cms.PSet(
    src = cms.InputTag("layer0MuonIsolations","muIsoDepositJets"),
    deltaR = cms.double(0.3)
    ) )
##store deposits
allLayer1Muons.isoDeposits = cms.PSet(
    tracker = cms.InputTag("layer0MuonIsolations", "muIsoDepositTk"),
    ecal = cms.InputTag("layer0MuonIsolations", "muIsoDepositCalByAssociatorTowersecal"),
    hcal = cms.InputTag("layer0MuonIsolations", "muIsoDepositCalByAssociatorTowershcal"),
    user = cms.VInputTag(
    cms.InputTag("layer0MuonIsolations", "muIsoDepositCalByAssociatorTowersho"),
    cms.InputTag("layer0MuonIsolations", "muIsoDepositJets") )
    )


#---------------------------------------
# Tau
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi import allLayer1Taus

## input source
allLayer1Taus.tauSource = cms.InputTag("allLayer0Taus")
## embed AOD objects?
allLayer1Taus.embedLeadTrack = cms.bool(False)
allLayer1Taus.embedSignalTracks = cms.bool(False)
allLayer1Taus.embedIsolationTracks = cms.bool(False)
## mc matching
allLayer1Taus.addGenMatch = cms.bool(True)
allLayer1Taus.genParticleMatch = cms.InputTag("tauMatch")
## resolution
allLayer1Taus.addResolutions = cms.bool(True)
allLayer1Taus.useNNResolutions = cms.bool(True)
allLayer1Taus.tauResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_tau.root')


#---------------------------------------
# Jet
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import allLayer1Jets

## input source
allLayer1Jets.jetSource = cms.InputTag("allLayer0Jets")
## embed AOD objects?
allLayer1Jets.embedCaloTowers = cms.bool(True)
## jec factors
addJetCorrFactors = cms.bool(True)
jetCorrFactorsSource = cms.InputTag("layer0JetCorrFactors")
## jet flavour idetification configurables
allLayer1Jets.getJetMCFlavour = cms.bool(True)
allLayer1Jets.JetPartonMapSource = cms.InputTag("jetFlavourAssociation")
## mc matching
allLayer1Jets.addGenPartonMatch = cms.bool(True)
allLayer1Jets.genPartonMatch = cms.InputTag("jetPartonMatch")
allLayer1Jets.addGenJetMatch = cms.bool(True)
allLayer1Jets.genJetMatch    = cms.InputTag("jetGenJetMatch")
allLayer1Jets.addPartonJetMatch = cms.bool(False)
allLayer1Jets.partonJetSource   = cms.InputTag("NOT_IMPLEMENTED")
## resolution
allLayer1Jets.addResolutions   = cms.bool(True)
allLayer1Jets.useNNResolutions = cms.bool(False)
allLayer1Jets.caliJetResoFile  = cms.string('PhysicsTools/PatUtils/data/Resolutions_lJets_MCJetCorJetIcone5.root')
allLayer1Jets.caliBJetResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_bJets_MCJetCorJetIcone5.root')
## b tag
allLayer1Jets.addBTagInfo = cms.bool(True)
allLayer1Jets.addDiscriminators   = cms.bool(True)
allLayer1Jets.discriminatorModule = cms.InputTag("layer0BTags")
allLayer1Jets.discriminatorNames  = cms.vstring('*')
allLayer1Jets.addTagInfoRefs = cms.bool(True)
allLayer1Jets.tagInfoModule = cms.InputTag("layer0TagInfos")
allLayer1Jets.tagInfoNames = cms.vstring('secondaryVertexTagInfos',
                                         'softElectronTagInfos',
                                         'softMuonTagInfos',
                                         'impactParameterTagInfos')
## track association
allLayer1Jets.addAssociatedTracks = cms.bool(True)
allLayer1Jets.trackAssociationSource = cms.InputTag("layer0JetTracksAssociator")
## jet charge
allLayer1Jets.addJetCharge = cms.bool(True)
allLayer1Jets.jetChargeSource = cms.InputTag("layer0JetCharge")


#---------------------------------------
# MET
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import allLayer1METs

## input source
allLayer1METs.metSource = cms.InputTag("allLayer0METs")
## mc matching
allLayer1METs.addGenMET = cms.bool(True)
allLayer1METs.genMETSource = cms.InputTag("genMet")
## resolution
allLayer1METs.addResolutions   = cms.bool(True)
allLayer1METs.useNNResolutions = cms.bool(False)
allLayer1METs.metResoFile = cms.string('PhysicsTools/PatUtils/data/Resolutions_met.root')
