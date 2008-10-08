import FWCore.ParameterSet.Config as cms

#---------------------------------------
# Electron
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import allLayer1Electrons

allLayer1Electrons.electronSource    = "allLayer0Electrons"                  ## input source
allLayer1Electrons.embedTrack        = False                                 ## embed AOD objects?
allLayer1Electrons.embedGsfTrack     = True    
allLayer1Electrons.embedSuperCluster = True
allLayer1Electrons.addGenMatch       = True                                  ## mc matching
allLayer1Electrons.embedGenMatch     = True
allLayer1Electrons.genParticleMatch  = "electronMatch"
allLayer1Electrons.addResolutions    = True                                  ## resolution
allLayer1Electrons.useNNResolutions  = False
allLayer1Electrons.electronResoFile  = 'PhysicsTools/PatUtils/data/Resolutions_electron.root'
allLayer1Electrons.isolation.tracker = cms.PSet(                             ## isolation for tracker
    src    = cms.InputTag("layer0ElectronIsolations", "eleIsoDepositTk"),
    deltaR = cms.double(0.3)
    )
allLayer1Electrons.isolation.ecal    = cms.PSet(                             ## isolation for ecal
    src    = cms.InputTag("layer0ElectronIsolations", "eleIsoDepositEcalFromClusts"),
   #src    = cms.InputTag("layer0ElectronIsolations","eleIsoDepositEcalFromHits"),
    deltaR = cms.double(0.4)
    )
allLayer1Electrons.isolation.hcal    = cms.PSet(                             ## isolation for hcal
    src    = cms.InputTag("layer0ElectronIsolations", "eleIsoDepositHcalFromTowers"),
    deltaR = cms.double(0.4)
    )
allLayer1Electrons.isoDeposits       = cms.PSet(                             ## store deposits
    tracker= cms.InputTag("layer0ElectronIsolations", "eleIsoDepositTk"),
    ecal   = cms.InputTag("layer0ElectronIsolations", "eleIsoDepositEcalFromClusts"),
   #ecal   = cms.InputTag("layer0ElectronIsolations","eleIsoDepositEcalFromHits"),
    hcal   = cms.InputTag("layer0ElectronIsolations", "eleIsoDepositHcalFromTowers")
    )
allLayer1Electrons.userData.userFunctions      = ['pt() / (pt() + trackIso() + caloIso())']
allLayer1Electrons.userData.userFunctionLabels = ['relIso']

#---------------------------------------
# Muon
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import allLayer1Muons

allLayer1Muons.muonSource            = "allLayer0Muons"                      ## input source
allLayer1Muons.embedTrack            = True                                  ## embed AOD objects?
allLayer1Muons.embedCombinedMuon     = True      
allLayer1Muons.embedStandAloneMuon   = True
allLayer1Muons.addGenMatch           = True                                  ## mc matching
allLayer1Muons.embedGenMatch         = True  
allLayer1Muons.genParticleMatch      = "muonMatch"
allLayer1Muons.addResolutions        = True                                  ## resolution
allLayer1Muons.useNNResolutions      = False
allLayer1Muons.muonResoFile          = 'PhysicsTools/PatUtils/data/Resolutions_muon.root'
allLayer1Muons.isolation.tracker     = cms.PSet(                             ## isolation for tracker
    src    = cms.InputTag("layer0MuonIsolations", "muIsoDepositTk"),
    deltaR = cms.double(0.3)
    )
allLayer1Muons.isolation.ecal        = cms.PSet(                             ## isolation for ecal
    src    = cms.InputTag("layer0MuonIsolations", "muIsoDepositCalByAssociatorTowersecal"),
    deltaR = cms.double(0.3)
    )
allLayer1Muons.isolation.hcal        = cms.PSet(                             ## isolation for hcal
    src    = cms.InputTag("layer0MuonIsolations", "muIsoDepositCalByAssociatorTowershcal"),
    deltaR = cms.double(0.3)
    )
allLayer1Muons.isolation.user        = cms.VPSet(                            ## isolation for ho
      cms.PSet(
      src    = cms.InputTag("layer0MuonIsolations", "muIsoDepositCalByAssociatorTowersho"),
      deltaR = cms.double(0.3)
      ),
      cms.PSet(
      src = cms.InputTag("layer0MuonIsolations","muIsoDepositJets"),
      deltaR = cms.double(0.3)
      )
    )
allLayer1Muons.isoDeposits           = cms.PSet(                             ## store deposits
    tracker = cms.InputTag("layer0MuonIsolations", "muIsoDepositTk"),
    ecal    = cms.InputTag("layer0MuonIsolations", "muIsoDepositCalByAssociatorTowersecal"),
    hcal    = cms.InputTag("layer0MuonIsolations", "muIsoDepositCalByAssociatorTowershcal"),
    user    = cms.VInputTag(
      cms.InputTag("layer0MuonIsolations", "muIsoDepositCalByAssociatorTowersho"),
      cms.InputTag("layer0MuonIsolations", "muIsoDepositJets")
      )
    )
allLayer1Muons.userData.userFunctions      = cms.vstring('pt() / (pt() + trackIso() + caloIso())')
allLayer1Muons.userData.userFunctionLabels = cms.vstring('relIso')

#---------------------------------------
# Tau
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi import allLayer1Taus

allLayer1Taus.tauSource              = "allLayer0Taus"                       ## input source
allLayer1Taus.embedLeadTrack         = True                                  ## embed AOD objects?
allLayer1Taus.embedSignalTracks      = True      
allLayer1Taus.embedIsolationTracks   = True
allLayer1Taus.addGenMatch            = True                                  ## mc matching
allLayer1Taus.embedGenMatch          = True            
allLayer1Taus.genParticleMatch       = "tauMatch"
allLayer1Taus.addResolutions         = True                                  ## resolution
allLayer1Taus.useNNResolutions       = True
allLayer1Taus.tauResoFile            = 'PhysicsTools/PatUtils/data/Resolutions_tau.root'


#---------------------------------------
# Jet
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import allLayer1Jets

allLayer1Jets.jetSource              = "allLayer0Jets"                       ## input source
allLayer1Jets.embedCaloTowers        = False                                 ## embed AOD objects?
allLayer1Jets.addJetCorrFactors      = True                                  ## jec factors
allLayer1Jets.jetCorrFactorsSource   = "layer0JetCorrFactors"
allLayer1Jets.getJetMCFlavour        = True                                  ## jet flavour idetification
allLayer1Jets.JetPartonMapSource     = "jetFlavourAssociation"
allLayer1Jets.addGenPartonMatch      = True                                  ## mc matching
allLayer1Jets.genPartonMatch         = "jetPartonMatch"
allLayer1Jets.addGenJetMatch         = True
allLayer1Jets.genJetMatch            = "jetGenJetMatch"
allLayer1Jets.addPartonJetMatch      = False
allLayer1Jets.partonJetSource        = "NOT_IMPLEMENTED"
allLayer1Jets.addResolutions         = True                                  ## resolution
allLayer1Jets.useNNResolutions       = False
allLayer1Jets.caliJetResoFile        = 'PhysicsTools/PatUtils/data/Resolutions_lJets_MCJetCorJetIcone5.root'
allLayer1Jets.caliBJetResoFile       = 'PhysicsTools/PatUtils/data/Resolutions_bJets_MCJetCorJetIcone5.root'
allLayer1Jets.addBTagInfo            = True                                  ## b tag
allLayer1Jets.addDiscriminators      = True
allLayer1Jets.discriminatorModule    = "layer0BTags"
allLayer1Jets.discriminatorNames     = '*'
allLayer1Jets.addTagInfoRefs         = True
allLayer1Jets.tagInfoModule          = "layer0TagInfos"
allLayer1Jets.tagInfoNames           = ['secondaryVertexTagInfos',
                                        'softElectronTagInfos',
                                        'softMuonTagInfos',
                                        'impactParameterTagInfos']
allLayer1Jets.addAssociatedTracks    = True                                  ## track association
allLayer1Jets.trackAssociationSource = "layer0JetTracksAssociator"
allLayer1Jets.addJetCharge           = True                                  ## jet charge
allLayer1Jets.jetChargeSource        = "layer0JetCharge"


#---------------------------------------
# MET
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import allLayer1METs

allLayer1METs.metSource              = "allLayer0METs"                       ## input source
allLayer1METs.addGenMET              = True
allLayer1METs.genMETSource           = "genMet"                              ## mc matching
allLayer1METs.addResolutions         = True                                  ## resolution
allLayer1METs.useNNResolutions       = False
allLayer1METs.metResoFile            = 'PhysicsTools/PatUtils/data/Resolutions_met.root'
