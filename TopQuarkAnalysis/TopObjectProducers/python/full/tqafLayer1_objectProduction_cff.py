import FWCore.ParameterSet.Config as cms

#
# keep potential top spaecific default replacements here
#

#---------------------------------------
# Electron
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import allLayer1Electrons

## to used from PhysicsTools/PatAlgos V04-14-03 onwards
## this mirrors the patTuple defaulkts for the summer08
## production at the moment
allLayer1Electrons.addTrigMatch  = True
allLayer1Electrons.trigPrimMatch = cms.VInputTag(
    cms.InputTag("electronTrigMatchHLTIsoEle15LWL1I"), 
    cms.InputTag("electronTrigMatchHLTEle15LWL1R"), 
    cms.InputTag("electronTrigMatchHLTDoubleIsoEle10LWL1I"), 
    cms.InputTag("electronTrigMatchHLTDoubleEle5SWL1R")
    )

#---------------------------------------
# Muon
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import allLayer1Muons

## to used from PhysicsTools/PatAlgos V04-14-03 onwards
## this mirrors the patTuple defaulkts for the summer08
## production at the moment
allLayer1Muons.addTrigMatch  = True
allLayer1Muons.trigPrimMatch = cms.VInputTag(
    cms.InputTag("muonTrigMatchHLTIsoMu11"), 
    cms.InputTag("muonTrigMatchHLTMu11"), 
    cms.InputTag("muonTrigMatchHLTDoubleIsoMu3"), 
    cms.InputTag("muonTrigMatchHLTDoubleMu3")
    )

#---------------------------------------
# Tau
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi import allLayer1Taus

## to used from PhysicsTools/PatAlgos V04-14-03 onwards
## this mirrors the patTuple defaulkts for the summer08
## production at the moment
allLayer1Taus.addTrigMatch  = True
allLayer1Taus.trigPrimMatch = cms.VInputTag(
    cms.InputTag("tauTrigMatchHLTLooseIsoTauMET30L1MET"), 
    cms.InputTag("tauTrigMatchHLTDoubleIsoTauTrk3")
    )

#---------------------------------------
# Photon
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cfi import allLayer1Photons

## to used from PhysicsTools/PatAlgos V04-14-03 onwards
## this mirrors the patTuple defaulkts for the summer08
## production at the moment
allLayer1Photons.addTrigMatch = False

#---------------------------------------
# Jet
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import allLayer1Jets

## to used from PhysicsTools/PatAlgos V04-14-03 onwards
## this mirrors the patTuple defaulkts for the summer08
## production at the moment
allLayer1Jets.addTrigMatch  = False

#---------------------------------------
# MET
#---------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import allLayer1METs

## to used from PhysicsTools/PatAlgos V04-14-03 onwards
## this mirrors the patTuple defaulkts for the summer08
## production at the moment
allLayer1METs.addTrigMatch = False
