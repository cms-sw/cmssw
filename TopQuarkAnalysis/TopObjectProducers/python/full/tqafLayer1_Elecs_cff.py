import FWCore.ParameterSet.Config as cms

#
# L1 input
#

## import module
from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import allLayer1Electrons

## configure for tqaf
allLayer1Electrons.electronSource        = 'allLayer0Electrons'
allLayer1Electrons.addGenMatch           = True
allLayer1Electrons.genParticleMatch      = 'electronMatch'
allLayer1Electrons.addResolutions        = True
allLayer1Electrons.useNNResolutions      = False
allLayer1Electrons.electronResoFile      = 'PhysicsTools/PatUtils/data/Resolutions_electron.root'
allLayer1Electrons.isolation.tracker.src = 'layer0ElectronIsolations:eleIsoDepositTk'
allLayer1Electrons.isolation.ecal.src    = 'layer0ElectronIsolations:eleIsoDepositEcalFromClusts'
allLayer1Electrons.isolation.hcal.src    = 'layer0ElectronIsolations:eleIsoDepositHcalFromTowers'
allLayer1Electrons.isolation.user        = []
allLayer1Electrons.isoDeposits.tracker   = 'layer0ElectronIsolations:eleIsoDepositTk'
allLayer1Electrons.isoDeposits.ecal      = 'layer0ElectronIsolations:eleIsoDepositEcalFromClusts'
allLayer1Electrons.isoDeposits.hcal      = 'layer0ElectronIsolations:eleIsoDepositHcalFromTowers'
allLayer1Electrons.addElectronID         = True

#
# L1 selection
#

## import module
from PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi import selectedLayer1Electrons

## configure for tqaf
selectedLayer1Electrons.src              = 'allLayer1Electrons'
selectedLayer1Electrons.cut              = 'pt > 10. & abs(eta) < 3.0'

