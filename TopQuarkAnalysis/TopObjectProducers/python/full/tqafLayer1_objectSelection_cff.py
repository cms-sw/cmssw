import FWCore.ParameterSet.Config as cms

#
# keep potential top spaecific default replacements here
#

#---------------------------------------
# Electron
#---------------------------------------
from PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi import selectedLayer1Electrons

selectedLayer1Electrons.src = cms.InputTag("allLayer1Electrons")
selectedLayer1Electrons.cut = cms.string('pt > 0.')

#---------------------------------------
# Muon
#---------------------------------------
from PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi     import selectedLayer1Muons

selectedLayer1Muons.src = cms.InputTag("allLayer1Muons")
selectedLayer1Muons.cut = cms.string('pt > 0.')


#---------------------------------------
# Tau
#---------------------------------------
from PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi      import selectedLayer1Taus

selectedLayer1Taus.src = cms.InputTag("allLayer1Taus")
selectedLayer1Taus.cut = cms.string('pt > 10.')


#---------------------------------------
# Jet
#---------------------------------------
from PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi      import selectedLayer1Jets

selectedLayer1Jets.src = cms.InputTag("allLayer1Jets")
selectedLayer1Jets.cut = cms.string('et > 15. & nConstituents > 0')


#---------------------------------------
# MET
#---------------------------------------
from PhysicsTools.PatAlgos.selectionLayer1.metSelector_cfi      import selectedLayer1METs

selectedLayer1METs.src = cms.InputTag("allLayer1METs")
selectedLayer1METs.cut = cms.string('et >= 0.')
