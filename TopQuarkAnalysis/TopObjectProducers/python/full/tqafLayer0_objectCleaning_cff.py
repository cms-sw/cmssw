import FWCore.ParameterSet.Config as cms

#---------------------------------------
# Electron
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.electronCleaner_cfi import allLayer0Electrons

## input source
allLayer0Electrons.electronSource = cms.InputTag("pixelMatchGsfElectrons")
## remove duplicates ?
allLayer0Electrons.removeDuplicates = cms.bool(True)
## select special ID
allLayer0Electrons.selection = cms.PSet( type = cms.string('none') )
## configure isolation for tracker
allLayer0Electrons.isolation.tracker = cms.PSet(
    src    = cms.InputTag("patAODElectronIsolations", "eleIsoDepositTk"),
    deltaR = cms.double(0.3),                                        ## POG suggestion
    cut    = cms.double(5.0)                                         ## isolation cut (as educated guess)
    )
## configure isolation for ecal
allLayer0Electrons.isolation.ecal = cms.PSet(
    src       = cms.InputTag("patAODElectronIsolations", "eleIsoDepositEcalFromClusts"),
    deltaR    = cms.double(0.4),                                     ## POG suggestion
    cut       = cms.double(5.0)                                      ## isolation cut (as educated guess)
    )
## configure isolation for hcal
allLayer0Electrons.isolation.hcal = cms.PSet(
    src       = cms.InputTag("patAODElectronIsolations", "eleIsoDepositHcalFromTowers"),
    deltaR    = cms.double(0.4),                                     ## POG suggestion
    cut       = cms.double(5.0)                                      ## isolation cut (as educated guess)
    )
## keep non isolated electrons in the event record
allLayer0Electrons.bitsToIgnore = cms.vstring('Isolation/All')       ## keep non isolated electrons (but flag them)


#---------------------------------------
# Muon
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.muonCleaner_cfi import allLayer0Muons

## source input
allLayer0Muons.muonSource = cms.InputTag("muons")
## select special ID
allLayer0Muons.selection = cms.PSet( type = cms.string('none') )
## configure isolation for tracker
allLayer0Muons.isolation.tracker = cms.PSet(
    src     = cms.InputTag("patAODMuonIsolations","muIsoDepositTk"), ##
    deltaR  = cms.double(0.3),                                       ## POG suggestion
    cut     = cms.double(2.0)                                        ## isolation cut (as educated guess)
    )
## configure isolation for ecal
allLayer0Muons.isolation.ecal = cms.PSet(
    src     = cms.InputTag("patAODMuonIsolations","muIsoDepositCalByAssociatorTowersecal"),
    deltaR  = cms.double(0.3),                                       ## POG suggestion
    cut     = cms.double(2.0)                                        ## isolation cut (as educated guess)
    )
## configure isolation for hcal
allLayer0Muons.isolation.hcal = cms.PSet(
    src = cms.InputTag("patAODMuonIsolations","muIsoDepositCalByAssociatorTowershcal"),
    deltaR  = cms.double(0.3),                                       ## POG suggestion
    cut     = cms.double(2.0)                                        ## isolation cut (as educated guess)
    )
allLayer0Muons.isolation.user = cms.VPSet(
    cms.PSet(
    src = cms.InputTag("patAODMuonIsolations","muIsoDepositCalByAssociatorTowersho"),
    deltaR = cms.double(0.3),
    cut = cms.double(2.0)
    ),
    cms.PSet(
    src = cms.InputTag("patAODMuonIsolations","muIsoDepositJets"),
    deltaR = cms.double(0.5),
    cut = cms.double(2.0)
    ) )
## keep non isolated muons in the event record
allLayer0Muons.bitsToIgnore = cms.vstring('Isolation/All')           ## keep non isolated muons (but flag them)


#---------------------------------------
# Tau
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.caloTauCleaner_cfi import allLayer0CaloTaus

## input source
allLayer0CaloTaus.tauSource = 'pfRecoTauProducer'
## discriminator source
allLayer0CaloTaus.tauDiscriminatorSource = 'pfRecoTauDiscriminationByIsolation'


#---------------------------------------
# Jet
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.caloJetCleaner_cfi import allLayer0Jets

## input source
allLayer0Jets.jetSource = cms.InputTag("iterativeCone5CaloJets")
## select special ID
allLayer0Jets.selection = cms.PSet( type = cms.string('none') )
## remove overlap with isolated electrons
allLayer0Jets.removeOverlaps = cms.PSet(
    electrons = cms.PSet(
    collection= cms.InputTag("allLayer0Electrons"), ##
    deltaR    = cms.double(0.3),                    ##
    cut       = cms.string('pt > 10'),              ## as in LeptonJetIsolationAngle
    flags     = cms.vstring('Isolation/Tracker'),   ## request the item to be marked as isolated in the tracker
                                                    ## by the PATElectronCleaner
    ) )


#---------------------------------------
# MET
#---------------------------------------
from PhysicsTools.PatAlgos.cleaningLayer0.caloMetCleaner_cfi import allLayer0METs

## input source
allLayer0METs.metSource = 'corMetType1Icone5Muons'
