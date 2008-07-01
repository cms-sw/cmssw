import FWCore.ParameterSet.Config as cms

countLayer1Leptons.electronSource = 'selectedLayer1Electrons'
countLayer1Leptons.muonSource = 'selectedLayer1Muons'
countLayer1Leptons.tauSource = 'selectedLayer1Taus'
countLayer1Leptons.countElectrons = True
countLayer1Leptons.countMuons = True
countLayer1Leptons.countTaus = False
countLayer1Leptons.minNumber = 1
countLayer1Leptons.maxNumber = 1
minLayer1Electrons.src = 'selectedLayer1Electrons'
minLayer1Electrons.minNumber = 0
maxLayer1Electrons.src = 'selectedLayer1Electrons'
maxLayer1Electrons.maxNumber = 999999
minLayer1Muons.src = 'selectedLayer1Muons'
minLayer1Muons.minNumber = 0
maxLayer1Muons.src = 'selectedLayer1Muons'
maxLayer1Muons.maxNumber = 999999
minLayer1Taus.src = 'selectedLayer1Taus'
minLayer1Taus.minNumber = 0
maxLayer1Taus.src = 'selectedLayer1Taus'
maxLayer1Taus.maxNumber = 999999

