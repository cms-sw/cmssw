import FWCore.ParameterSet.Config as cms

# PATLayer1 Electron input
allLayer1Electrons.electronSource = 'allLayer0Electrons'
allLayer1Electrons.addGenMatch = True
allLayer1Electrons.genParticleMatch = 'electronMatch'
allLayer1Electrons.addTrigMatch = False
# replace allLayer1Electrons.trigPrimMatch          = { electronTrigMatchHLT1ElectronRelaxed,
#                                                       electronTrigMatchCandHLT1ElectronStartup }
allLayer1Electrons.addResolutions = True
allLayer1Electrons.useNNResolutions = False
allLayer1Electrons.electronResoFile = 'PhysicsTools/PatUtils/data/Resolutions_electron.root'
allLayer1Electrons.tracksSource = 'generalTracks'
allLayer1Electrons.isolation.tracker.src = 'layer0ElectronIsolations:eleIsoDepositTk'
allLayer1Electrons.isolation.ecal.src = 'layer0ElectronIsolations:eleIsoDepositEcalFromClusts'
allLayer1Electrons.isolation.hcal.src = 'layer0ElectronIsolations:eleIsoDepositHcalFromTowers'
allLayer1Electrons.isolation.user = []
allLayer1Electrons.isoDeposits.tracker = 'layer0ElectronIsolations:eleIsoDepositTk'
allLayer1Electrons.isoDeposits.ecal = 'layer0ElectronIsolations:eleIsoDepositEcalFromClusts'
allLayer1Electrons.isoDeposits.hcal = 'layer0ElectronIsolations:eleIsoDepositHcalFromTowers'
allLayer1Electrons.addElectronID = True
# PATLayer1 Electron selection
selectedLayer1Electrons.src = 'allLayer1Electrons'
# replace selectedLayer1Electrons.cut  = "pt > 10. & abs(eta) < 2.4 & trackIso < 3 & caloIso < 6 & leptonID > 0.99"
selectedLayer1Electrons.cut = 'pt > 10. & abs(eta) < 2.4 & trackIso < 3 & caloIso < 6'

