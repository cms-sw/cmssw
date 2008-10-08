import FWCore.ParameterSet.Config as cms

#---------------------------------------
# Electron
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.electronCleaner_cfi import allLayer0Electrons

allLayer0Electrons.electronSource   = "pixelMatchGsfElectrons"        ## input source
allLayer0Electrons.removeDuplicates = True                            ## remove duplicates?
allLayer0Electrons.selection        = cms.PSet(                       ## select special ID
    type = cms.string('none')
    )
allLayer0Electrons.isolation.tracker= cms.PSet(                       ## configure isolation for tracker
    src    = cms.InputTag("patAODElectronIsolations", "eleIsoDepositTk"),
    deltaR = cms.double(0.3),                                         ## POG suggestion
    cut    = cms.double(5.0)                                          ## isolation cut (as educated guess)
    )
allLayer0Electrons.isolation.ecal   = cms.PSet(                       ## configure isolation for ecal
    src       = cms.InputTag("patAODElectronIsolations", "eleIsoDepositEcalFromClusts"),
   #src       = cms.InputTag("patAODElectronIsolations","eleIsoDepositEcalFromHits"), ## recommendation from POG
    deltaR    = cms.double(0.4),                                      ## POG suggestion
    cut       = cms.double(5.0)                                       ## isolation cut (as educated guess)
    )
allLayer0Electrons.isolation.hcal   = cms.PSet(                       ## configure isolation for hcal
    src       = cms.InputTag("patAODElectronIsolations", "eleIsoDepositHcalFromTowers"),
    deltaR    = cms.double(0.4),                                      ## POG suggestion
    cut       = cms.double(5.0)                                       ## isolation cut (as educated guess)
    )
allLayer0Electrons.bitsToIgnore     = ['Isolation/All']               ## keep non isolated electrons (but flag them)


#---------------------------------------
# Muon
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.muonCleaner_cfi import allLayer0Muons

allLayer0Muons.muonSource           = "muons"                         ## source input
allLayer0Muons.selection            = cms.PSet(                       ## select special ID
    type = cms.string('none')
    )
allLayer0Muons.isolation.tracker    = cms.PSet(                       ## configure isolation for tracker
    src     = cms.InputTag("patAODMuonIsolations","muIsoDepositTk"),  ##
    deltaR  = cms.double(0.3),                                        ## POG suggestion
    cut     = cms.double(2.0)                                         ## isolation cut (as educated guess)
    )
allLayer0Muons.isolation.ecal       = cms.PSet(                       ## configure isolation for ecal
    src     = cms.InputTag("patAODMuonIsolations","muIsoDepositCalByAssociatorTowersecal"),
    deltaR  = cms.double(0.3),                                        ## POG suggestion
    cut     = cms.double(2.0)                                         ## isolation cut (as educated guess)
    )
allLayer0Muons.isolation.hcal       = cms.PSet(                       ## configure isolation for hcal
    src = cms.InputTag("patAODMuonIsolations","muIsoDepositCalByAssociatorTowershcal"),
    deltaR  = cms.double(0.3),                                        ## POG suggestion
    cut     = cms.double(2.0)                                         ## isolation cut (as educated guess)
    )
allLayer0Muons.isolation.user       = cms.VPSet(
      cms.PSet(
      src = cms.InputTag("patAODMuonIsolations","muIsoDepositCalByAssociatorTowersho"),
      deltaR = cms.double(0.3),
      cut = cms.double(2.0)
      ),
      cms.PSet(
      src = cms.InputTag("patAODMuonIsolations","muIsoDepositJets"),
      deltaR = cms.double(0.5),
      cut = cms.double(2.0)
      )
    )
allLayer0Muons.bitsToIgnore         = ['Isolation/All']               ## keep non isolated muons (but flag them)


#---------------------------------------
# Tau
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.pfTauCleaner_cfi import allLayer0Taus

allLayer0Taus.tauSource                  = 'pfRecoTauProducer'                    ## input source
allLayer0Taus.tauDiscriminatorSource     = 'pfRecoTauDiscriminationByIsolation'   ## discriminator source

from PhysicsTools.PatAlgos.cleaningLayer0.caloTauCleaner_cfi import allLayer0CaloTaus

allLayer0CaloTaus.tauSource              = 'caloRecoTauProducer'                  ## input source
allLayer0CaloTaus.tauDiscriminatorSource = 'caloRecoTauDiscriminationByIsolation' ## discriminator source


#---------------------------------------
# Jet
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.caloJetCleaner_cfi import allLayer0Jets

allLayer0Jets.jetSource             = "iterativeCone5CaloJets"        ## input source
allLayer0Jets.selection             = cms.PSet(                       ## select special ID
    type = cms.string('none')
    )
allLayer0Jets.removeOverlaps        = cms.PSet(                       ## remove overlap with isolated electrons
    electrons = cms.PSet(
    collection= cms.InputTag("allLayer0Electrons"),                   ##
    deltaR    = cms.double(0.3),                                      ##
    cut       = cms.string('pt > 10'),                                ## as in LeptonJetIsolationAngle
    flags     = cms.vstring('Isolation/Tracker'),                     ## request the item to be marked as isolated in the tracker
                                                                      ## by the PATElectronCleaner
    ) )


#---------------------------------------
# MET
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.caloMetCleaner_cfi import allLayer0METs

allLayer0METs.metSource             = 'corMetType1Icone5Muons'        ## input source
