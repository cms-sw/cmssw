import FWCore.ParameterSet.Config as cms

#
# define derivations from std pat defaults
#

#-------------------------------------------------------------------------------------------
#
# production
#
#-------------------------------------------------------------------------------------------
from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import allLayer1Electrons

## to used from PhysicsTools/PatAlgos V04-14-03 onwards
allLayer1Electrons.addTrigMatch  = True
allLayer1Electrons.trigPrimMatch = cms.VInputTag(
    cms.InputTag("electronTrigMatchHLTIsoEle15LWL1I"), 
    cms.InputTag("electronTrigMatchHLTEle15LWL1R"), 
    cms.InputTag("electronTrigMatchHLTDoubleIsoEle10LWL1I"), 
    cms.InputTag("electronTrigMatchHLTDoubleEle5SWL1R")
    )

from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import allLayer1Muons

## to used from PhysicsTools/PatAlgos V04-14-03 onwards
allLayer1Muons.addTrigMatch  = True
allLayer1Muons.trigPrimMatch = cms.VInputTag(
    cms.InputTag("muonTrigMatchHLTIsoMu11"), 
    cms.InputTag("muonTrigMatchHLTMu11"), 
    cms.InputTag("muonTrigMatchHLTDoubleIsoMu3"), 
    cms.InputTag("muonTrigMatchHLTDoubleMu3")
    )

from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import allLayer1Jets

## to used from PhysicsTools/PatAlgos V04-14-03 onwards
allLayer1Jets.addTrigMatch  = False

from PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi import allLayer1Taus

## to used from PhysicsTools/PatAlgos V04-14-03 onwards
allLayer1Taus.addTrigMatch  = True
allLayer1Taus.trigPrimMatch = cms.VInputTag(
    cms.InputTag("tauTrigMatchHLTLooseIsoTauMET30L1MET"), 
    cms.InputTag("tauTrigMatchHLTDoubleIsoTauTrk3")
    )

from PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi import allLayer1METs

## to used from PhysicsTools/PatAlgos V04-14-03 onwards
allLayer1METs.addTrigMatch = False

from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cfi import allLayer1Photons

## to used from PhysicsTools/PatAlgos V04-14-03 onwards
allLayer1Photons.addTrigMatch = False

#
# selection
#
from PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi import selectedLayer1Electrons

selectedLayer1Electrons.src = cms.InputTag("allLayer1Electrons")
selectedLayer1Electrons.cut = cms.string('pt > 0.')

from PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi import selectedLayer1Muons

selectedLayer1Muons.src     = cms.InputTag("allLayer1Muons")
selectedLayer1Muons.cut     = cms.string('pt > 0.')

from PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi  import selectedLayer1Taus

selectedLayer1Taus.src      = cms.InputTag("allLayer1Taus")
selectedLayer1Taus.cut      = cms.string('pt > 10.')

from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi  import selectedLayer1Jets

selectedLayer1Jets.src      = cms.InputTag("allLayer1Jets")
selectedLayer1Jets.cut      = cms.string('et > 15. & nConstituents > 0')

from PhysicsTools.PatAlgos.selectionLayer1.metSelector_cfi  import selectedLayer1METs

selectedLayer1METs.src      = cms.InputTag("allLayer1METs")
selectedLayer1METs.cut      = cms.string('et >= 0.')
